# Modal Infrastructure for UMM Codebase

Run inference, training, and evaluation on cloud GPUs via [Modal](https://modal.com), with persistent model/dataset caching across runs.

## Setup

```bash
pip install modal
modal setup                # login (one-time)
```

Create required secrets on [Modal Dashboard](https://modal.com/secrets):

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
modal secret create wandb-secret WANDB_API_KEY=xxx       # for training
```

> **Note:** UEval, GEdit, ImgEdit, WISE, and DPG-Bench all share the **WISE image** for their scoring phase (local Qwen2.5-VL). Generation runs in the model's own image; scoring is dispatched to the WISE image via the `score_model: wise` field in the eval YAML.
iness


## Quick Start

```bash
# 1. Download model weights (one-time per model)
modal run modal/download.py --model bagel

# 2. Verify download
modal run modal/download.py --ls

# 3. Run inference
modal run modal/run.py --model bagel --cmd "python inferencer.py --model_path /model_cache/bagel/BAGEL-7B-MoT"

# 4. Run post-training
modal run modal/train.py --config bagel_sft

# 5. Sync local code changes to Modal
modal run modal/download.py --sync
```

## Commands

### download.py

```bash
modal run modal/download.py --model <name>     # bagel|blip3o|deepgen|emu3|emu3_5|janus_pro|janus_flow|mmada|omnigen2|ovis_u1|show_o|show_o2|tokenflow|post_train
modal run modal/download.py --dataset <name>   # ueval|uni_mmmu
modal run modal/download.py --all              # download everything
modal run modal/download.py --ls               # list cached contents
modal run modal/download.py --sync             # sync local codebase to Volume
```

### run.py

```bash
modal run modal/run.py --model <name>                        # verify environment
modal run modal/run.py --model <name> --script <file.py>     # run a script
modal run modal/run.py --model <name> --cmd "<command>"      # run shell command
modal run modal/run.py --model <name> --config <train_cfg>   # post-training via umm CLI
modal run modal/run.py --model <name> --eval-config <cfg>    # evaluation via umm CLI

# Use --gpu to select GPU type (default: A100-80GB), --detach to run in background
modal run --detach modal/run.py --model blip3o --eval-config modal_wise_blip3o --gpu H100
```

Supported models: `bagel`, `blip3o`, `deepgen`, `emu3`, `emu3_5`, `janus_pro`, `janus_flow`, `mmada`, `omnigen2`, `ovis_u1`, `show_o`, `show_o2`, `tokenflow`, `geneval`, `uni_mmmu`, `wise`

> Generation-only models use their own image. Scoring for UEval / GEdit / ImgEdit / WISE / DPG-Bench all runs in the **WISE image** (local Qwen2.5-VL), selected automatically via `score_model: wise` in the eval YAML — so there is no standalone `ueval` model to pass here.

#### Example: run WISE eval on Bagel (two-phase)

```bash
# Phase 1 — generate images in the Bagel image
modal run --detach modal/run.py --model bagel \
    --eval-config wise/wise_bagel_generate --gpu H100

# Phase 2 — score in the WISE image (Qwen2.5-VL)
modal run --detach modal/run.py --model wise \
    --eval-config wise/wise_bagel_score --gpu A100-80GB
```

Or run both phases from a single combined config — `run.py` reads `score_model: wise` from the YAML and dispatches the scoring phase to the WISE image automatically:

```bash
modal run --detach modal/run.py --model bagel \
    --eval-config wise/wise_bagel --gpu H100
```

## Volumes

| Volume | Mount Path | Purpose |
|--------|-----------|---------|
| `umm-model-cache` | `/model_cache` | HuggingFace model weights |
| `umm-post-train-model-cache` | `/post_train_model_cache` | Post-train model weights |
| `umm-datasets-cache` | `/datasets` | HuggingFace datasets |
| `umm-checkpoints` | `/checkpoints` | Training checkpoints |
| `umm-outputs` | `/outputs` | Inference & eval outputs |

```bash
modal volume ls umm-model-cache               # list contents
modal volume get umm-outputs results/ ./out/   # download to local
modal volume rm umm-outputs old_run/           # delete
```

## Flash Attention Setup

| Model(s) | Python | CUDA | PyTorch | flash-attn | Status |
|---|---|---|---|---|---|
| Janus-Pro, Show-o2, MMaDA, Ovis-U1, WISE | 3.10 | 12.4 | 2.5.1 | 2.7.4 | Required |
| Bagel | 3.10 | 12.4 | 2.5.1 | 2.5.8 | Required |
| BLIP3-o | 3.11 | 12.1 | 2.3.0 | 2.6.2 | Required |
| Emu3 | 3.10 | 12.1 | 2.2.1 | 2.5.7 | Required |
| Emu3.5 | 3.12 | 12.4 | 2.8.0 | 2.8.3 | Required |
| OmniGen2 | 3.11 | 12.4 | 2.6.0 | 2.7.4 | Recommended |
| DeepGen | 3.10 | 12.4 | 2.8.0 | latest | Recommended |

> Models sharing a row use the exact same pre-compiled flash-attn wheel (`flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`).

Models **not** using flash-attn: Janus (1.3B), JanusFlow, Show-o (v1), TokenFlow.

## Adding a New Model

1. Add HF repo ID in `config.py` → `HF_MODELS`
2. Add container image in `images.py`
3. Add repo directory mapping in `run.py` → `repo_dirs`
4. Download: `modal run modal/download.py --model <name>`
