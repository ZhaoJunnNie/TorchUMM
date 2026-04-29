#!/usr/bin/env python3
"""Submit MM-Vet prediction JSON to the official HuggingFace Space evaluator.

The official evaluator is a Gradio app:
https://huggingface.co/spaces/whyu/MM-Vet_Evaluator

Use --dry-run first to verify the local file and remote API schema without
starting the full GPT-based grading queue.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


SPACE_URL = "https://whyu-mm-vet-evaluator.hf.space"


def _load_predictions(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"MM-Vet prediction file must be a JSON object: {path}")
    if not data:
        raise ValueError(f"MM-Vet prediction file is empty: {path}")
    bad_keys = [k for k in data if not str(k).startswith("v1_")]
    if bad_keys:
        preview = ", ".join(map(str, bad_keys[:5]))
        raise ValueError(f"MM-Vet keys should look like 'v1_0'; bad keys: {preview}")
    return data


def _client():
    try:
        from gradio_client import Client
    except ImportError as exc:
        raise RuntimeError("gradio_client is required. Install or use an env that provides it.") from exc
    return Client(SPACE_URL)


def dry_run(path: Path) -> int:
    data = _load_predictions(path)
    client = _client()
    api = client.view_api(return_format="dict")
    endpoints = set(api.get("named_endpoints", {}).keys())
    if "/run_grade" not in endpoints:
        raise RuntimeError(f"Official MM-Vet Space does not expose /run_grade; endpoints={sorted(endpoints)}")
    print(f"[mmvet official] local JSON OK: {path}")
    print(f"[mmvet official] predictions: {len(data)}")
    print(f"[mmvet official] remote API OK: {SPACE_URL} /run_grade")
    return 0


def submit(path: Path, out: Path | None, model: str, api_key: str, api_base: str) -> int:
    _load_predictions(path)
    try:
        from gradio_client import handle_file
    except ImportError as exc:
        raise RuntimeError("gradio_client is required. Install or use an env that provides it.") from exc

    client = _client()
    print(f"[mmvet official] submitting {path} to {SPACE_URL} with model={model}", flush=True)
    result = client.predict(handle_file(str(path)), api_key, model, api_base, api_name="/run_grade")
    result_path = Path(result)
    if out is None:
        out = path.with_name(path.stem + "_official_mmvet_score.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result_path, out)
    print(f"[mmvet official] wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="MM-Vet prediction JSON produced by TorchUMM.")
    parser.add_argument("--out", default=None, help="Output zip path. Defaults next to --json.")
    parser.add_argument("--model", default="gpt-4.1", choices=["gpt-4.1", "gpt-4-0613", "gpt-4-turbo"])
    parser.add_argument("--api-key", default="", help="Required for gpt-4-0613/gpt-4-turbo. gpt-4.1 uses the Space key.")
    parser.add_argument("--api-base", default="", help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--dry-run", action="store_true", help="Validate local file and remote API without scoring.")
    args = parser.parse_args()

    pred_path = Path(args.json).expanduser()
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    out = Path(args.out).expanduser() if args.out else None
    if args.dry_run:
        return dry_run(pred_path)
    return submit(pred_path, out, args.model, args.api_key, args.api_base)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[mmvet official] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
