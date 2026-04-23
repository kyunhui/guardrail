from __future__ import annotations

import os
import re
from typing import List

import torch
from PIL import Image

from src.classifier_models.base import (
    PromptHarmfulness,
    ResponseHarmfulness,
    SafetyClassifierBase,
    SafetyClassifierOutput,
)


class LlamaGuard3Vision11B(SafetyClassifierBase):
    """
    Llama Guard 3 Vision 11B safety classifier.

    This classifier follows the same input/output contract as LlamaGuard2/3:
    - required input fields: ["prompt"]
    - optional input fields: ["response"]
    - output fields: ["prompt_harmfulness", "response_harmfulness"]

    For prompt harmfulness, it classifies with a single user turn.
    For response harmfulness, it classifies on the (user, assistant) conversation.
    """

    HF_MODEL_ID = "meta-llama/Llama-Guard-3-11B-Vision"

    def __init__(self, batch_size: int = 64, **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)
        self.max_new_tokens = int(kwargs.pop("max_new_tokens", 16))
        self.max_sequence_length = int(kwargs.pop("max_sequence_length", 8192))
        self.model_name_or_path = kwargs.pop("local_model_path", None) or type(self).HF_MODEL_ID
        self._torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        self.model = None
        self.processor = None

    def get_required_input_fields(self) -> list[str]:
        return ["prompt"]

    def get_optional_input_fields(self) -> list[str]:
        return ["response", "image_path"]

    def get_output_fields(self) -> list[str]:
        return ["prompt_harmfulness", "response_harmfulness"]

    def _lazy_load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        from transformers import AutoProcessor, MllamaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            device_map="auto",
            torch_dtype=self._torch_dtype,
        ).eval()

        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token_id is None and self.processor.tokenizer.eos_token is not None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    @staticmethod
    def _parse_safe_unsafe(decoded_text: str) -> str | None:
        text = decoded_text.strip()
        if not text:
            return None
        first_line = text.split("\n")[0].strip().lower()
        if re.match(r"unsafe\b", first_line):
            return "unsafe"
        if re.match(r"safe\b", first_line):
            return "safe"
        return None

    def _build_conversations(self, items: list[dict[str, str]], include_response: bool) -> list[list[dict]]:
        conversations: list[list[dict]] = []
        for item in items:
            user_content: list[dict] = []
            image_path = item.get("image_path")
            if image_path and os.path.isfile(image_path):
                user_content.append({"type": "image", "image": image_path})
            user_content.append({"type": "text", "text": item["prompt"]})
            turns: list[dict] = [{"role": "user", "content": user_content}]
            if include_response:
                turns.append(
                    {"role": "assistant", "content": [{"type": "text", "text": item.get("response", "")}]}
                )
            conversations.append(turns)
        return conversations

    @torch.inference_mode()
    def _classify_conversations(self, conversations: list[list[dict]]) -> tuple[list[float], list[float]]:
        self._lazy_load()
        assert self.model is not None and self.processor is not None

        def conversation_has_image(conv: list[dict]) -> bool:
            for turn in conv:
                content = turn.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "image":
                            p = c.get("image")
                            if isinstance(p, str) and os.path.isfile(p):
                                return True
            return False

        # Mllama processor requires image-per-sample consistency within a batch.
        # If mixed, run in two sub-batches and stitch results back in original order.
        has_image_mask = [conversation_has_image(conv) for conv in conversations]
        if any(has_image_mask) and not all(has_image_mask):
            safe_probs: list[float] = [0.5] * len(conversations)
            unsafe_probs: list[float] = [0.5] * len(conversations)

            idx_with = [i for i, v in enumerate(has_image_mask) if v]
            idx_without = [i for i, v in enumerate(has_image_mask) if not v]

            conv_with = [conversations[i] for i in idx_with]
            conv_without = [conversations[i] for i in idx_without]

            safe_with, unsafe_with = self._classify_conversations(conv_with)
            safe_without, unsafe_without = self._classify_conversations(conv_without)

            for i, s, u in zip(idx_with, safe_with, unsafe_with):
                safe_probs[i] = s
                unsafe_probs[i] = u
            for i, s, u in zip(idx_without, safe_without, unsafe_without):
                safe_probs[i] = s
                unsafe_probs[i] = u
            return safe_probs, unsafe_probs

        images: list[list[Image.Image]] = []
        has_any_image = False
        for conv in conversations:
            conv_imgs: list[Image.Image] = []
            for turn in conv:
                content = turn.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "image":
                            p = c.get("image")
                            if isinstance(p, str) and os.path.isfile(p):
                                conv_imgs.append(Image.open(p).convert("RGB"))
                                has_any_image = True
            images.append(conv_imgs)

        formatted_inputs: List[str] = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        if has_any_image:
            encoded_inputs = self.processor(
                text=formatted_inputs,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_sequence_length,
            ).to("cuda")
        else:
            encoded_inputs = self.processor(
                text=formatted_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_sequence_length,
            ).to("cuda")

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None) if hasattr(self.processor, "tokenizer") else None
        eos_id = getattr(self.model.generation_config, "eos_token_id", None)
        if pad_id is None and hasattr(self.processor, "tokenizer"):
            pad_id = self.processor.tokenizer.eos_token_id

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id

        gen_ids = self.model.generate(**encoded_inputs, **gen_kwargs)
        prompt_width = encoded_inputs["input_ids"].shape[1]
        new_ids = gen_ids[:, prompt_width:]

        safe_probs: list[float] = []
        unsafe_probs: list[float] = []
        for row in new_ids:
            row_list = row.tolist()
            if pad_id is not None:
                while row_list and row_list[-1] in (pad_id, eos_id):
                    row_list.pop()
            decoded = self.processor.tokenizer.decode(row_list, skip_special_tokens=True)  # type: ignore
            label = self._parse_safe_unsafe(decoded)
            if label == "unsafe":
                safe_probs.append(0.01)
                unsafe_probs.append(0.99)
            elif label == "safe":
                safe_probs.append(0.99)
                unsafe_probs.append(0.01)
            else:
                safe_probs.append(0.5)
                unsafe_probs.append(0.5)
        return safe_probs, unsafe_probs

    @torch.inference_mode()
    def _classify_batch(self, items: list[dict[str, str]]) -> list[SafetyClassifierOutput]:
        has_response = "response" in items[0]
        assert all("response" in item for item in items) or not any(
            "response" in item for item in items
        ), "response must be included in all or none of items"

        prompt_conversations = self._build_conversations(items, include_response=False)
        prompt_safe_probs, prompt_unsafe_probs = self._classify_conversations(prompt_conversations)

        if has_response:
            response_conversations = self._build_conversations(items, include_response=True)
            response_safe_probs, response_unsafe_probs = self._classify_conversations(response_conversations)
        else:
            response_safe_probs = [None] * len(items)
            response_unsafe_probs = [None] * len(items)

        outputs: list[SafetyClassifierOutput] = []
        for p_safe, p_unsafe, r_safe, r_unsafe in zip(
            prompt_safe_probs, prompt_unsafe_probs, response_safe_probs, response_unsafe_probs
        ):
            prompt_harmfulness = (
                PromptHarmfulness.HARMFUL if p_unsafe > p_safe else PromptHarmfulness.UNHARMFUL
            )
            metadata = {
                "prompt_safe_prob": p_safe,
                "prompt_unsafe_prob": p_unsafe,
            }

            if r_safe is not None and r_unsafe is not None:
                response_harmfulness = (
                    ResponseHarmfulness.HARMFUL if r_unsafe > r_safe else ResponseHarmfulness.UNHARMFUL
                )
                metadata["response_safe_prob"] = r_safe
                metadata["response_unsafe_prob"] = r_unsafe
            else:
                response_harmfulness = None

            outputs.append(
                SafetyClassifierOutput(
                    prompt_harmfulness=prompt_harmfulness,
                    response_harmfulness=response_harmfulness,
                    metadata=metadata,
                )
            )

        return outputs

