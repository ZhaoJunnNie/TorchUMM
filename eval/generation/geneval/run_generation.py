#!/usr/bin/env python3
"""
GenEval generation script — generates images for each GenEval prompt using
umm's InferencePipeline.

Called via subprocess from the geneval.py CLI wrapper.

Output layout (compatible with GenEval evaluate_images.py):
    images_dir/
        00000/
            metadata.jsonl
            samples/
                00000.png
                00001.png
                ...
        00001/
            ...

Usage:
    python eval/generation/geneval/run_generation.py --config configs/eval/geneval/geneval_bagel.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from umm.core.config import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(path_str: str, repo_root: Path) -> Path:
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


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_prompts(metadata_path: Path, max_samples: int) -> List[Dict[str, Any]]:
    """Load GenEval prompts from evaluation_metadata.jsonl."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"GenEval metadata not found: {metadata_path}")
    prompts: List[Dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if max_samples > 0:
        prompts = prompts[:max_samples]
    return prompts


# ---------------------------------------------------------------------------
# Image extraction helpers
# ---------------------------------------------------------------------------

def _extract_saved_paths(result: Any, fallback_dir: Path) -> List[str]:
    """Return generated image paths from a pipeline result."""
    from PIL import Image

    saved_paths: List[str] = []

    if isinstance(result, dict):
        for key in ("saved_paths", "output_path", "image_path", "image_paths"):
            val = result.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item:
                        p = Path(item)
                        if p.is_file():
                            saved_paths.append(str(p))
                if saved_paths:
                    return saved_paths
            if isinstance(val, str) and val:
                p = Path(val)
                if p.is_file():
                    return [str(p)]
        img = result.get("image")
        if isinstance(img, Image.Image):
            out_path = fallback_dir / "generated.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out_path), format="PNG")
            return [str(out_path)]
        imgs = result.get("images")
        if isinstance(imgs, list) and imgs:
            for idx, item in enumerate(imgs):
                if isinstance(item, str) and item:
                    p = Path(item)
                    if p.is_file():
                        saved_paths.append(str(p))
                elif isinstance(item, Image.Image):
                    out_path = fallback_dir / f"generated_{idx:05d}.png"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    item.save(str(out_path), format="PNG")
                    saved_paths.append(str(out_path))
            if saved_paths:
                return saved_paths
    if isinstance(result, Image.Image):
        out_path = fallback_dir / "generated.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(out_path), format="PNG")
        return [str(out_path)]

    # Scan fallback_dir for recently created images
    if fallback_dir.is_dir():
        img_exts = {".png", ".jpg", ".jpeg", ".webp"}
        candidates = [
            str(f) for f in sorted(
            [f for f in fallback_dir.rglob("*") if f.is_file() and f.suffix.lower() in img_exts],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        ]
        if candidates:
            return candidates
    return []


def _clear_workspace(workspace: Path) -> None:
    """Remove all image files from *workspace* so the fallback scan is unambiguous."""
    img_exts = {".png", ".jpg", ".jpeg", ".webp"}
    for f in workspace.rglob("*"):
        if f.is_file() and f.suffix.lower() in img_exts:
            f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

# Backbones that need seed passed explicitly in params
_SEED_IN_PARAMS_BACKBONES = {"omnigen2"}


def _distributed_context() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _run_generation(
    pipeline: Any,
    backbone: str,
    prompts: List[Dict[str, Any]],
    images_dir: Path,
    request_params: Dict[str, Any],
    images_per_prompt: int,
    resume: bool,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    """Generate images for each GenEval prompt in the expected directory structure."""
    images_dir.mkdir(parents=True, exist_ok=True)
    workspace = images_dir / f"_gen_workspace_rank{rank:02d}"
    workspace.mkdir(parents=True, exist_ok=True)

    total = len(prompts)
    n_ok = n_skip = n_err = 0

    base_seed = int(request_params.get("seed", 42))

    assigned = [(idx, metadata) for idx, metadata in enumerate(prompts) if idx % world_size == rank]

    for idx, metadata in tqdm(
        assigned,
        total=len(assigned),
        desc=f"[geneval gen rank {rank}]",
    ):
        prompt_dir = images_dir / f"{idx:05d}"
        samples_dir = prompt_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata for evaluate_images.py
        meta_file = prompt_dir / "metadata.jsonl"
        if not meta_file.exists():
            meta_file.write_text(
                json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
            )

        existing_samples = sorted(samples_dir.glob("*.png"))
        if resume and len(existing_samples) >= images_per_prompt:
            n_skip += 1
            continue

        prompt_text = metadata.get("prompt", "")
        prompt_ok = True
        sample_count = len(existing_samples) if resume else 0

        while sample_count < images_per_prompt:
            _clear_workspace(workspace)
            remaining = images_per_prompt - sample_count
            expected_path = workspace / "sample.png"

            if backbone in _SEED_IN_PARAMS_BACKBONES:
                sample_params = dict(request_params)
                sample_params["seed"] = base_seed + idx * images_per_prompt + sample_count
            else:
                sample_params = request_params

            payload: Dict[str, Any] = {
                "backbone": backbone,
                "task": "generation",
                "prompt": prompt_text,
                "output_path": str(expected_path),
                "params": sample_params,
            }
            try:
                result = pipeline.run(payload)
            except Exception as exc:
                print(f"[geneval] prompt={idx} sample={sample_count} generation error: {exc}")
                traceback.print_exc()
                prompt_ok = False
                break

            produced = [str(expected_path)] if expected_path.is_file() else _extract_saved_paths(result, workspace)
            produced = [p for p in produced if p and Path(p).is_file()]

            if not produced:
                print(f"[geneval] prompt={idx} sample={sample_count}: no image produced")
                prompt_ok = False
                break

            for saved in produced[:remaining]:
                out_file = samples_dir / f"{sample_count:05d}.png"
                shutil.copy2(saved, str(out_file))
                sample_count += 1

        if prompt_ok:
            n_ok += 1
        else:
            n_err += 1

    return {
        "total": total,
        "assigned": len(assigned),
        "ok": n_ok,
        "skipped": n_skip,
        "error": n_err,
    }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_eval_cfg(
    config_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    raw_cfg = load_config(config_path)
    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    geneval_cfg = (
        raw_cfg.get("geneval", {})
        if isinstance(raw_cfg.get("geneval"), dict)
        else {}
    )
    inference_cfg = (
        raw_cfg.get("inference", {})
        if isinstance(raw_cfg.get("inference"), dict)
        else {}
    )
    return eval_cfg, geneval_cfg, inference_cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="GenEval generation (image)")
    parser.add_argument("--config", required=True, help="Path to eval config YAML")
    args = parser.parse_args()

    _eval_cfg, geneval_cfg, inference_cfg = _load_eval_cfg(args.config)
    # eval/generation/geneval/run_generation.py -> parents[3] = repo root
    repo_root = Path(__file__).resolve().parents[3]
    rank, world_size, local_rank = _distributed_context()

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # --- Resolve inference config (support infer_config reference) ---
    infer_config_ref = inference_cfg.get("infer_config")
    if isinstance(infer_config_ref, str) and infer_config_ref:
        infer_cfg_path = _resolve_path(infer_config_ref, repo_root)
        resolved_cfg = load_config(infer_cfg_path)
        resolved_inf = resolved_cfg.get("inference", resolved_cfg)
        if isinstance(resolved_inf, dict):
            inference_cfg = resolved_inf

    # --- Backbone ---
    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError(
            "`inference.backbone` is required for GenEval generation. "
            "Either set it directly or via `inference.infer_config`."
        )
    backbone = _normalize_backbone_name(backbone_raw)

    backbone_cfg = inference_cfg.get("backbone_cfg", {})
    if not isinstance(backbone_cfg, dict):
        backbone_cfg = {}

    request_cfg = inference_cfg.get("request", {})
    request_params: Dict[str, Any] = {}
    if isinstance(request_cfg, dict):
        params = request_cfg.get("params", {})
        if isinstance(params, dict):
            request_params = dict(params)

    # --- GenEval config ---
    metadata_path_val = geneval_cfg.get("metadata_path")
    if metadata_path_val:
        metadata_path = _resolve_path(str(metadata_path_val), repo_root)
    else:
        # Default: use the metadata bundled in prompts/
        metadata_path = Path(__file__).resolve().parent / "prompts" / "evaluation_metadata.jsonl"

    out_dir = _resolve_path(
        str(geneval_cfg.get("out_dir", f"output/geneval/{backbone}")), repo_root
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = out_dir / "images"
    max_samples = int(geneval_cfg.get("max_samples", 0) or 0)
    images_per_prompt = int(geneval_cfg.get("images_per_prompt", 4) or 4)
    resume = bool(geneval_cfg.get("resume", True))

    print(
        f"[geneval] backbone={backbone}, out_dir={out_dir}, "
        f"images_per_prompt={images_per_prompt}, "
        f"max_samples={max_samples or 'all'}, resume={resume}, "
        f"rank={rank}, world_size={world_size}"
    )

    # --- Load prompts ---
    prompts = _load_prompts(metadata_path, max_samples)
    print(f"[geneval] loaded {len(prompts)} prompts")

    # --- Run generation ---
    from umm.inference import InferencePipeline

    pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

    gen_summary = _run_generation(
        pipeline=pipeline,
        backbone=backbone,
        prompts=prompts,
        images_dir=images_dir,
        request_params=request_params,
        images_per_prompt=images_per_prompt,
        resume=resume,
        rank=rank,
        world_size=world_size,
    )

    print(
        f"[geneval] generation done — "
        f"ok={gen_summary['ok']}, skipped={gen_summary['skipped']}, error={gen_summary['error']}"
    )

    # Save generation summary
    if rank == 0:
        summary_path = out_dir / "gen_summary.json"
        summary_path.write_text(json.dumps(gen_summary, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
