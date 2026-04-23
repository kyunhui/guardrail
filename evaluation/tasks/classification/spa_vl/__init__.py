import abc
import json
import os
import re
from collections import defaultdict

from evaluation.schemas import ClassificationTaskBase
from evaluation.tasks.metrics import get_average, get_f1
from evaluation.tasks.classification._hf_image_cache import ensure_cached_image
from src.classifier_models.base import (
    PromptHarmfulness,
    ResponseHarmfulness,
    SafetyClassifierBase,
)

DEFAULT_SPA_VL_JSON_PATH = os.path.join(
    os.path.dirname(__file__),
    "SPA_VL_Eval.json",
)
DEFAULT_SPA_VL_IMAGE_ROOT = os.path.join(
    os.path.dirname(__file__),
    "spa_eval_label1_cache",
)

_REQUEST_RE = re.compile(r"Request:\s*(harmful|unharmful)", re.IGNORECASE)
_RESPONSE_RE = re.compile(r"Response:\s*(harmful|unharmful)", re.IGNORECASE)
_HUMAN_RE = re.compile(r"Human user:\n(.*?)\n\nAI assistant:\n", re.IGNORECASE | re.DOTALL)
_ASSISTANT_RE = re.compile(r"AI assistant:\n(.*)", re.IGNORECASE | re.DOTALL)


def _to_prompt_label(token: str) -> PromptHarmfulness:
    return PromptHarmfulness.HARMFUL if token.lower() == "harmful" else PromptHarmfulness.UNHARMFUL


def _to_response_label(token: str) -> ResponseHarmfulness:
    return ResponseHarmfulness.HARMFUL if token.lower() == "harmful" else ResponseHarmfulness.UNHARMFUL


def _parse_prompt_and_response(user_text: str) -> tuple[str, str]:
    m_h = _HUMAN_RE.search(user_text)
    m_a = _ASSISTANT_RE.search(user_text)
    if not m_h or not m_a:
        return user_text.strip(), ""
    prompt = m_h.group(1).replace("<image>", "").strip()
    response = m_a.group(1).strip()
    return prompt, response


def _parse_labels(assistant_text: str) -> tuple[PromptHarmfulness, ResponseHarmfulness]:
    m_req = _REQUEST_RE.search(assistant_text)
    m_resp = _RESPONSE_RE.search(assistant_text)
    if not m_req or not m_resp:
        raise ValueError("Could not parse Request/Response labels from assistant content.")
    return _to_prompt_label(m_req.group(1)), _to_response_label(m_resp.group(1))


def _resolve_image_path(image_rel: str | None, image_root: str) -> str | None:
    if not image_rel:
        return None
    # 1) try direct basename first
    base = os.path.basename(image_rel)
    direct = os.path.join(image_root, base)
    if os.path.isfile(direct):
        return direct

    # 2) SPA cache naming fallback: "./SPA_VL_Eval/12.jpg" -> "000012.png"
    stem, _ = os.path.splitext(base)
    if stem.isdigit():
        idx = int(stem)
        fallback = os.path.join(image_root, f"{idx:06d}.png")
        if os.path.isfile(fallback):
            return fallback

    # 3) unresolved
    return None


class SPAVLBase(ClassificationTaskBase, abc.ABC):
    def __init__(
        self,
        dataset_json_path: str = DEFAULT_SPA_VL_JSON_PATH,
        image_root: str = DEFAULT_SPA_VL_IMAGE_ROOT,
        **kwargs,
    ):
        self.dataset_json_path = dataset_json_path
        self.image_root = image_root
        super().__init__(**kwargs)

    def _load_raw(self) -> list[dict]:
        if not os.path.isfile(self.dataset_json_path):
            raise FileNotFoundError(f"SPA-VL json not found: {self.dataset_json_path}")
        with open(self.dataset_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Images must match JSON row order, not `./SPA_VL_Eval/{stem}.jpg` names:
        # GuardReasoner-VL `generate.py` uses `SPA_VL_Eval['test'][i]['image']` with the
        # same loop index `i` as this JSON. Local caches store `000000.png` … in that order.
        out: list[dict] = []
        download_attempted = 0
        download_succeeded = 0
        for row_index, row in enumerate(data):
            messages = row.get("messages", [])
            if len(messages) < 2:
                continue
            user_text = str(messages[0].get("content", ""))
            assistant_text = str(messages[1].get("content", ""))
            prompt, response = _parse_prompt_and_response(user_text)
            prompt_label, response_label = _parse_labels(assistant_text)

            image_rel = None
            image_path = None
            images = row.get("images", [])
            if isinstance(images, list) and images:
                image_rel = images[0]
                sequential = os.path.join(self.image_root, f"{row_index:06d}.png")
                if os.path.isfile(sequential):
                    image_path = sequential
                else:
                    download_attempted += 1
                    if download_attempted == 1:
                        print(
                            "[spa_vl] Missing cached images detected. "
                            "Downloading from Hugging Face dataset..."
                        )
                    image_path = ensure_cached_image(
                        image_root=self.image_root,
                        remote_relative_path=f"test/SPA_VL_Eval/{row_index:04d}.jpg",
                        local_filename=f"{row_index:04d}.jpg",
                    )
                    if image_path is not None:
                        download_succeeded += 1
                    if download_attempted % 100 == 0:
                        print(
                            f"[spa_vl] Download progress: attempted={download_attempted}, "
                            f"succeeded={download_succeeded}"
                        )
                    if image_path is None:
                        image_path = _resolve_image_path(image_rel, self.image_root)

            out.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "prompt_harmfulness": prompt_label,
                    "response_harmfulness": response_label,
                    "image_rel": image_rel,
                    "image_path": image_path,
                }
            )
        if not out:
            raise ValueError("No valid rows loaded for SPA-VL.")
        if download_attempted > 0:
            print(
                f"[spa_vl] Download complete: attempted={download_attempted}, "
                f"succeeded={download_succeeded}"
            )
        return out


class SPAVLPromptSafetyEval(SPAVLBase):
    def required_input_fields(self) -> list[str]:
        return ["prompt"]

    @property
    def label_field(self) -> str:
        return "prompt_harmfulness"

    def required_output_fields(self) -> list[list[str]]:
        return [["prompt_harmfulness"]]

    def load(self) -> list[dict]:
        # Keep response/image fields for debugging and model compatibility.
        return self._load_raw()

    def _evaluate(self, model: SafetyClassifierBase) -> tuple[dict, list[dict]]:
        classification_results = model.classify(self.data)
        pred_is_correct = defaultdict(list)
        individual_results = []
        for gt, pred in zip(self.data, classification_results):
            gt_label = gt[self.label_field]
            individual_results.append(
                {
                    "id": gt["id"],
                    "prompt": gt["prompt"],
                    "response": gt["response"],
                    "image_path": gt.get("image_path"),
                    "gt_prompt_harmfulness": gt_label.value,
                    **pred.asdict(),
                }
            )
            if pred.is_parsing_error:
                pred_is_correct["parsing_error"].append(1)
                pred_is_correct["micro_acc"].append(0)
                gt_label_is_harmful = gt_label == PromptHarmfulness.HARMFUL
                pred_is_correct["f1"].append(
                    {"pred": not gt_label_is_harmful, "gt": gt_label_is_harmful}
                )
                pred_is_correct[gt_label.value].append(0)
            else:
                pred_is_correct["parsing_error"].append(0)
                pred_is_correct["micro_acc"].append(gt_label == pred.prompt_harmfulness)
                pred_is_correct["f1"].append(
                    {
                        "pred": pred.prompt_harmfulness == PromptHarmfulness.HARMFUL,
                        "gt": gt_label == PromptHarmfulness.HARMFUL,
                    }
                )
                pred_is_correct[gt_label.value].append(gt_label == pred.prompt_harmfulness)

        return {
            k: get_average(v) if "f1" not in k else get_f1(v) for k, v in pred_is_correct.items()
        }, individual_results


class SPAVLResponseSafetyEval(SPAVLBase):
    def required_input_fields(self) -> list[str]:
        return ["prompt", "response"]

    @property
    def label_field(self) -> str:
        return "response_harmfulness"

    def required_output_fields(self) -> list[list[str]]:
        return [["response_harmfulness"]]

    def load(self) -> list[dict]:
        return self._load_raw()

    def _evaluate(self, model: SafetyClassifierBase) -> tuple[dict, list[dict]]:
        classification_results = model.classify(self.data)
        pred_is_correct = defaultdict(list)
        individual_results = []
        for gt, pred in zip(self.data, classification_results):
            gt_label = gt[self.label_field]
            individual_results.append(
                {
                    "id": gt["id"],
                    "prompt": gt["prompt"],
                    "response": gt["response"],
                    "image_path": gt.get("image_path"),
                    "gt_response_harmfulness": gt_label.value,
                    **pred.asdict(),
                }
            )
            if pred.is_parsing_error:
                pred_is_correct["parsing_error"].append(1)
                pred_is_correct["micro_acc"].append(0)
                gt_label_is_harmful = gt_label == ResponseHarmfulness.HARMFUL
                pred_is_correct["f1"].append(
                    {"pred": not gt_label_is_harmful, "gt": gt_label_is_harmful}
                )
                pred_is_correct[gt_label.value].append(0)
            else:
                pred_is_correct["parsing_error"].append(0)
                pred_is_correct["micro_acc"].append(gt_label == pred.response_harmfulness)
                pred_is_correct["f1"].append(
                    {
                        "pred": pred.response_harmfulness == ResponseHarmfulness.HARMFUL,
                        "gt": gt_label == ResponseHarmfulness.HARMFUL,
                    }
                )
                pred_is_correct[gt_label.value].append(gt_label == pred.response_harmfulness)

        return {
            k: get_average(v) if "f1" not in k else get_f1(v) for k, v in pred_is_correct.items()
        }, individual_results

