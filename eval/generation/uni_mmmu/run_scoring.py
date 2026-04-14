#!/usr/bin/env python3
"""Uni-MMMU scoring-only script.

Usage:
    python eval/generation/uni_mmmu/run_scoring.py --config configs/eval/uni_mmmu/emu3.yaml

Calls eval_ummmu.py with the appropriate --base_path and --model_name
derived from the UMM YAML config.  This script can be run in a separate
Python environment that has the scoring dependencies installed
(transformers, qwen_vl_utils, dreamsim, cairosvg, etc.).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from umm.core.config import load_config


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    # Expand environment variables (e.g. ${UMM_MODEL_CACHE}) before resolving
    path_str = os.path.expandvars(path_str)
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def _normalize_backbone_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "showo2": "show_o2",
        "showo": "show_o2",
        "janus": "janus_pro",
        "januspro": "janus_pro",
        "omnigen": "omnigen2",
        "blip3": "blip3o",
        "blip3_o": "blip3o",
        "token_flow": "tokenflow",
    }
    return aliases.get(normalized, normalized)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Uni-MMMU scoring by invoking eval_ummmu.py."
    )
    parser.add_argument(
        "--config", required=True,
        help="UMM YAML config containing `inference` and `uni_mmmu` blocks.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    raw_cfg = load_config(args.config)

    inference_cfg = raw_cfg.get("inference", {})
    if not isinstance(inference_cfg, dict):
        inference_cfg = {}

    uni_mmmu_cfg = raw_cfg.get("uni_mmmu", {})
    if not isinstance(uni_mmmu_cfg, dict):
        uni_mmmu_cfg = {}

    backbone_raw = inference_cfg.get("backbone", "")
    backbone = _normalize_backbone_name(str(backbone_raw))

    out_dir = _resolve_path(
        str(uni_mmmu_cfg.get("out_dir", f"output/uni_mmmu/{backbone}")),
        repo_root,
    )

    scoring_cfg = uni_mmmu_cfg.get("scoring", {})
    if not isinstance(scoring_cfg, dict):
        scoring_cfg = {}

    data_root_value = uni_mmmu_cfg.get("data_root")
    if not data_root_value:
        print("[scoring] ERROR: `uni_mmmu.data_root` is required.")
        return 1
    data_root = str(_resolve_path(str(data_root_value), repo_root))

    # Generation writes to: {out_dir}/{task}/  (e.g. /outputs/uni_mmmu/bagel/math/)
    # We use --outputs_path and --eval_path to directly point eval_ummmu.py
    # at the correct directories, avoiding the base_path/outputs/{model_name}
    # path construction which would add an extra "outputs/" level.
    model_name = out_dir.name
    eval_output = str(out_dir.parent / "eval" / model_name)

    eval_script = Path(__file__).resolve().parent / "eval_ummmu.py"
    if not eval_script.exists():
        print(f"[scoring] ERROR: eval_ummmu.py not found at {eval_script}")
        return 1

    cmd = [
        sys.executable,
        str(eval_script),
        "--model_name", model_name,
        "--base_path", str(out_dir.parent),
        "--outputs_path", str(out_dir),
        "--eval_path", eval_output,
        "--data_root", data_root,
    ]

    max_items = scoring_cfg.get("max_items_per_task")
    if max_items is not None:
        cmd.extend(["--max_items", str(int(max_items))])

    qwen3 = scoring_cfg.get("text_lm_model") or scoring_cfg.get("qwen3_model")
    if qwen3:
        cmd.extend(["--qwen3_model", os.path.expandvars(str(qwen3))])

    qwen_vl = scoring_cfg.get("vl_model") or scoring_cfg.get("qwen2_5_vl_model")
    if qwen_vl:
        cmd.extend(["--qwen2_5_vl_model", os.path.expandvars(str(qwen_vl))])

    vl_attn = scoring_cfg.get("vl_attn_impl")
    if vl_attn:
        cmd.extend(["--vl_attn_impl", str(vl_attn)])

    dreamsim_cache = scoring_cfg.get("dreamsim_cache")
    if dreamsim_cache:
        cmd.extend(["--dreamsim_cache", os.path.expandvars(str(dreamsim_cache))])

    if raw_cfg.get("uni_mmmu", {}).get("resume"):
        cmd.append("--resume")

    print(f"[scoring] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_root)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
