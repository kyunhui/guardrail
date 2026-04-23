# Ai2 Safety Tool 🧰 (Classification Evaluation Suite)

This repository contains code for safety classifier evaluation across prompt harmfulness, response harmfulness, and response refusal tasks.

- WildGuard: Open One-stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs
  - <a href="https://arxiv.org/abs/2406.18495"><img src="https://img.shields.io/badge/📝-paper-blue"></a> <a href="https://github.com/allenai/wildguard"><img src="https://img.shields.io/badge/🔗-code-red"></a> <a href="https://huggingface.co/allenai/wildguard"><img src="https://img.shields.io/badge/🤗-wildguard (model)-green"></a> <a href="https://huggingface.co/datasets/allenai/wildguardmix"><img src="https://img.shields.io/badge/🤗-wildguardmix (data)-orange">
  </a>

This lets you compare classifier performance across text-only and multimodal safety benchmarks.

<img width="610" alt="image" src="https://github.com/user-attachments/assets/7d29f65f-ab6a-4164-8adc-0c1dc860bb30">

  
## Updates

- 2025-10-16: add support for BBQ, StrongReject, WMDP
- 2025-08-29: add support for reasoning models.
- 2024-07-06: add support for MMLU, TruthfulQA, and sorrybench classifier.

## Features

- Easy evaluation of **safety classifiers** on three tasks (detection of _prompt harmfulness_, _response harmfulness_, and _response refusal_) across 10+ benchmarks.
- Easy addition of new benchmarks and models to the evaluation suite.

## Installation

```bash
conda create -n guardrail python=3.11 && conda activate guardrail
pip install -e .
pip install -r requirements.txt
```

## Quick Start (Recommended Scripts)

Use the helper scripts in the repo root for consistent setup and execution.

### 1) Create `venv_gr` and install dependencies

```bash
cd /path/to/guardrail
bash scripts/setup_venv_gr.sh
```

This script:
- creates `venv_gr` if it does not exist,
- activates it,
- upgrades `pip/setuptools/wheel`,
- installs dependencies from `requirements.txt`.

### 2) Run classifier evaluation with `run_eval.sh`

```bash
cd /path/to/guardrail
bash scripts/run_eval.sh \
  --model_name WildGuard \
  --tasks wildguardtest_prompt,wildguardtest_response,wildguardtest_refusal,openai_mod \
  --report_output_path ./classification_results/metrics.json \
  --save_individual_results_path ./classification_results/all.json
```

To see argument help:

```bash
bash scripts/run_eval.sh
```

## _Safety Classifier_ Evaluation

### Prompt harmfulness benchmarks

- [WildGuardTest](https://arxiv.org/abs/2406.18495)
- [ToxicChat](https://arxiv.org/abs/2310.17389)
- [OpenAI Moderation](https://ojs.aaai.org/index.php/AAAI/article/view/26752)
- [AegisSafetyTest](https://arxiv.org/abs/2404.05993)
- [SimpleSafetyTests](https://arxiv.org/abs/2311.08370)
- [Harmbench Prompt](https://arxiv.org/abs/2402.04249)
- HarmImageTest (image-only prompt harmfulness)
- SPA-VL Eval (multimodal prompt harmfulness)
  
### Response harmfulness benchmarks

- [WildGuardTest](https://arxiv.org/abs/2406.18495)
- [Harmbench Response](https://arxiv.org/abs/2402.04249)
- [SafeRLHF](https://arxiv.org/abs/2406.15513)
- [BeaverTails](https://arxiv.org/abs/2307.04657)
- [XSTest-Resp](https://arxiv.org/abs/2406.18495)
- SPA-VL Eval (multimodal response harmfulness)

### Response refusal benchmarks

- [WildGuardTest](https://arxiv.org/abs/2406.18495)
- [XSTest-Resp](https://arxiv.org/abs/2406.18495)

### How-to-use

The commands below allow for running benchmarks to evaluate quality of safety classifiers such as WildGuard and LlamaGuard. The first command can be used to run all included benchmarks, while the second can be used to run select benchmarks.
To specify a task, the syntax is `<folder>:<config_yaml>`,
where `folder` is a folder under `tasks/classification` and `config_yaml` is the name of the configuration yaml file excluding `.yaml`.

```

# run all classification benchmarks by a single command

export CUDA_VISIBLE_DEVICES={NUM_GPUS};
python evaluation/run_all_classification_benchmarks.py \
    --model_name WildGuard \
    --report_output_path ./classification_results/metrics.json \
    --save_individual_results_path ./classification_results/all.json

# run specific classification benchmarks by a single command. here, we use four benchmarks

python evaluation/eval.py classifiers \
  --model_name WildGuard \
  --tasks wildguardtest_prompt,wildguardtest_response,wildguardtest_refusal,openai_mod \
  --report_output_path ./classification_results/metrics.json \
  --save_individual_results_path ./classification_results/all.json

```

Equivalent script form:

```bash
bash scripts/run_eval.sh \
  --model_name WildGuard \
  --tasks wildguardtest_prompt,wildguardtest_response,wildguardtest_refusal,openai_mod \
  --report_output_path ./classification_results/metrics.json \
  --save_individual_results_path ./classification_results/all.json
```

### GuardReasoner-VL-style full benchmark set (prompt/response + multimodal)

If you want to run the exact set below:

- Prompt: `toxicchat`, `openai_mod`, `simplesafetytests`, `harmbench`, `wildguardtest_prompt`, `harmimage`, `spa_vl`
- Response: `harmbench:response`, `saferlhf`, `beavertails`, `xstest_response_harm`, `wildguardtest_response`, `spa_vl:response`

use `evaluation/eval.py classifiers` with an explicit task list:

```bash
cd /path/to/guardrail
export CUDA_VISIBLE_DEVICES=0

python evaluation/eval.py classifiers \
  --model_name Qwen25VLInstruct \
  --tasks toxicchat,openai_mod,simplesafetytests,harmbench,wildguardtest_prompt,harmimage,spa_vl,harmbench:response,saferlhf,beavertails,xstest_response_harm,wildguardtest_response,spa_vl:response \
  --report_output_path ./classification_results/guardreasoner_vl_metrics.json \
  --save_individual_results_path ./classification_results/guardreasoner_vl_all.json \
  --override_existing_report true
```

Notes:

- `run_all_classification_benchmarks.py` currently covers text safety benchmarks only; it does not include `harmimage` or `spa_vl`.
- For VLM classifiers (e.g., `Qwen25VLInstruct`, `LlamaGuard3Vision11B`), use a model that supports `image_path` input.
- The multimodal tasks expect local image caches:
  - `harmimage`: `/root/workspace/GuardReasoner-VL/data/benchmark/spa_eval_label0_cache`
  - `spa_vl`: `/root/workspace/GuardReasoner-VL/data/benchmark/spa_eval_label1_cache`
- `spa_vl` supports both prompt and response runs:
  - prompt: `spa_vl`
  - response: `spa_vl:response`

## Acknowledgements

This repository uses some code from the:
- [Harmbench](https://github.com/centerforaisafety/HarmBench) -- in particular, code for model input templates,
- [StrongReject](https://github.com/dsbowen/strong_reject) -- in particular, code for logit analysis on the StrongReject benchmark

## Citation

```
@misc{wildguard2024,
      title={WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs}, 
      author={Seungju Han and Kavel Rao and Allyson Ettinger and Liwei Jiang and Bill Yuchen Lin and Nathan Lambert and Yejin Choi and Nouha Dziri},
      year={2024},
      eprint={2406.18495},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.18495}, 
}
```
