"""Run upstream lmms-eval tasks through a loaded UMM understanding pipeline.

Imports of lmms-eval and model backbones remain lazy so this optional bridge
does not add dependencies to the other ``umm`` commands.
"""
from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from umm.core.config import load_config


LMMS_EVAL_REVISION = "8f40097532a7bf4db6123713f1a905e02c8f5971"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else _REPO_ROOT / path).resolve()


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def _load_lmms_runtime(settings: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Import the selected checkout and report its observed source identity."""
    source_value = settings.get("source_path")
    expected_revision = settings.get("revision")
    source_root = _resolve_path(str(source_value)) if source_value else None
    if source_root is not None:
        if not (source_root / "lmms_eval" / "evaluator.py").is_file():
            raise FileNotFoundError(f"Not an lmms-eval checkout: {source_root}")
        sys.path.insert(0, str(source_root))
    try:
        package = importlib.import_module("lmms_eval")
        package_file = Path(package.__file__).resolve()
        observed_root = package_file.parent.parent
        if source_root is not None and observed_root != source_root:
            raise RuntimeError(
                f"lmms_eval is already imported from {observed_root}, expected {source_root}. "
                "Start a fresh Python process with the requested checkout."
            )
        try:
            revision = _git_output(observed_root, "rev-parse", "HEAD")
            changed = _git_output(
                observed_root, "status", "--porcelain", "--untracked-files=all", "--", "lmms_eval"
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            revision, changed = None, None
        if expected_revision and revision != str(expected_revision):
            raise RuntimeError(
                f"lmms-eval revision mismatch: expected {expected_revision}, observed "
                f"{revision or 'unknown (not a verifiable Git checkout)'} at {observed_root}."
            )
        if expected_revision and changed:
            raise RuntimeError(
                f"Pinned lmms-eval package has local changes at {observed_root}; "
                "use an unchanged checkout so the task/scorer identity is verifiable."
            )
        evaluator = importlib.import_module("lmms_eval.evaluator")
        evaluator_file = Path(evaluator.__file__).resolve()
        if evaluator_file != observed_root / "lmms_eval" / "evaluator.py":
            raise RuntimeError(f"Unexpected lmms-eval evaluator import: {evaluator_file}")
        from umm.eval.lmms_bridge import UMMGenerationAdapter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The lmms bridge needs the selected lmms-eval checkout and its dependencies "
            f"in this Python environment (missing module: {exc.name})."
        ) from exc
    return evaluator.simple_evaluate, UMMGenerationAdapter, {
        "package_path": str(package_file),
        "evaluator_path": str(evaluator_file),
        "git_revision": revision,
        "expected_revision": expected_revision,
        "package_changes": changed,
        "pinned_and_clean": bool(expected_revision and revision == expected_revision and not changed),
        "package_version": getattr(package, "__version__", None),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, partial):
        return {"callable": _json_default(value.func), "args": value.args, "keywords": value.keywords}
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{name}"
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"`{name}` must be a mapping.")
    return dict(value)


def run_lmms_evaluation(
    config: str | Path | Mapping[str, Any],
    *,
    pipeline: Any = None,
    model_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one pipeline without rebuilding an injected quantized model.

    ``model_identity`` is a caller-declared provenance record, not verification
    that a quantization intervention is active. Scientific callers should include
    their arm/checkpoint/quantization manifest and verify activation separately.
    The output directory must be new for every run.
    """
    config_path = str(Path(config).resolve()) if isinstance(config, (str, Path)) else None
    raw = load_config(config) if config_path else dict(config)
    eval_cfg = _mapping(raw.get("eval", {}), "eval")
    if str(eval_cfg.get("benchmark", "")).lower().strip() != "lmms":
        raise ValueError("Expected `eval.benchmark: lmms`.")
    settings = _mapping(raw.get("lmms", {}), "lmms")
    inference = _mapping(raw.get("inference", {}), "inference")
    tasks = settings.get("tasks")
    if not isinstance(tasks, list) or not tasks or any(not isinstance(t, str) or not t.strip() for t in tasks):
        raise ValueError("`lmms.tasks` must be a nonempty list of exact upstream task names.")
    if len(tasks) != len(set(tasks)):
        raise ValueError("`lmms.tasks` must not contain duplicate task names.")
    task_overrides = _mapping(settings.get("task_overrides", {}), "lmms.task_overrides")
    for task, value in task_overrides.items():
        if task not in tasks:
            raise ValueError(f"Task override refers to an unrequested task: {task}")
        override = _mapping(value, f"lmms.task_overrides.{task}")
        unsupported = set(override) - {"dataset_path", "dataset_kwargs"}
        if unsupported:
            raise ValueError(f"Task overrides may only change dataset_path/dataset_kwargs, got {sorted(unsupported)}")
        if "dataset_path" in override and not isinstance(override["dataset_path"], str):
            raise ValueError(f"`lmms.task_overrides.{task}.dataset_path` must be a string.")
        if "dataset_kwargs" in override:
            _mapping(override["dataset_kwargs"], f"lmms.task_overrides.{task}.dataset_kwargs")
        task_overrides[task] = override
    task_specs = [{"task": name, **task_overrides[name]} if name in task_overrides else name for name in tasks]
    bootstrap_iters = settings.get("bootstrap_iters", 100000)
    if isinstance(bootstrap_iters, bool) or not isinstance(bootstrap_iters, int) or bootstrap_iters < 0:
        raise ValueError("`lmms.bootstrap_iters` must be a nonnegative integer.")
    if int(settings.get("batch_size", 1)) != 1:
        raise ValueError("The initial UMM lmms bridge supports batch_size=1.")
    if not settings.get("out_dir"):
        raise ValueError("`lmms.out_dir` is required and must name a fresh run directory.")
    out_dir = _resolve_path(str(settings["out_dir"]))
    if out_dir.exists():
        raise FileExistsError(f"Refusing to reuse lmms output directory: {out_dir}")
    defaults = _mapping(settings.get("generation_defaults", {"max_new_tokens": 128}), "lmms.generation_defaults")
    overrides = _mapping(settings.get("generation_overrides", {}), "lmms.generation_overrides")
    identity = _mapping(model_identity or settings.get("model_identity", {}), "lmms.model_identity")
    if not identity:
        raise ValueError("Provide `model_identity` (or `lmms.model_identity`) for the evaluated state.")
    limit = settings.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit <= 0):
        raise ValueError("`lmms.limit` must be null or a positive integer/fraction.")
    if isinstance(limit, float) and not 0 < limit < 1:
        raise ValueError("A floating-point `lmms.limit` must be between zero and one.")
    evaluate, adapter_class, source = _load_lmms_runtime(settings)
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "benchmark": "lmms",
        "status": "initializing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "source": source,
        "tasks": tasks,
        "task_overrides": task_overrides,
        "model_identity": identity,
        "model_identity_is_caller_declared": True,
        "pipeline_injected": pipeline is not None,
        "inference_config": inference,
        "generation_defaults": defaults,
        "generation_overrides": overrides,
        "generation_precedence": "explicit overrides > upstream task kwargs > fallback defaults",
        "limit": limit,
        "batch_size": 1,
        "bootstrap_iters": bootstrap_iters,
        "response_cache": None,
        "request_cache": False,
        "out_dir": str(out_dir),
        "scoring_policy": "unchanged upstream tasks/scorers; raw responses retained",
        "limited_ocrbench_note": (
            "Upstream OCRBench divides by 1000 even with limit; its limited score is not subset accuracy."
            if limit is not None and "ocrbench" in tasks else None
        ),
    }
    _write_json(out_dir / "config.json", raw)
    _write_json(out_dir / "manifest.json", manifest)
    try:
        if pipeline is None:
            backbone = inference.get("backbone")
            if not isinstance(backbone, str) or not backbone.strip():
                raise ValueError("`inference.backbone` is required when no pipeline is injected.")
            backbone_cfg = _mapping(inference.get("backbone_cfg", {}), "inference.backbone_cfg")
            from umm.inference import InferencePipeline

            pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)
        manifest["pipeline_type"] = f"{type(pipeline).__module__}.{type(pipeline).__qualname__}"
        adapter = adapter_class(
            pipeline=pipeline,
            raw_output_path=out_dir / "generations.jsonl",
            image_dir=out_dir / "images",
            generation_defaults=defaults,
            generation_overrides=overrides,
            model_identity=identity,
        )
        manifest["status"] = "running"
        _write_json(out_dir / "manifest.json", manifest)
        seed = int(settings.get("seed", 0))
        result = evaluate(
            model=adapter,
            tasks=task_specs,
            batch_size=1,
            use_cache=None,
            cache_requests=False,
            limit=limit,
            log_samples=True,
            bootstrap_iters=bootstrap_iters,
            cli_args=SimpleNamespace(
                output_path=str(out_dir),
                process_with_media=False,
                reasoning_tags=None,
            ),
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
        )
        if not isinstance(result, dict):
            raise RuntimeError("lmms-eval returned no result; this bridge expects a single coordinator process.")
        samples = result.pop("samples", {})
        sample_paths: dict[str, str] = {}
        if samples:
            sample_dir = out_dir / "samples"
            sample_dir.mkdir()
            for task, rows in samples.items():
                filename = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task)) + ".jsonl"
                sample_path = sample_dir / filename
                with sample_path.open("x", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
                sample_paths[str(task)] = str(sample_path)
        _write_json(out_dir / "results.json", result)
        manifest.update({
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "sample_paths": sample_paths,
            "results_path": str(out_dir / "results.json"),
            "raw_generations_path": str(out_dir / "generations.jsonl"),
        })
        _write_json(out_dir / "manifest.json", manifest)
        return {"results": result, "manifest": manifest}
    except Exception as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        _write_json(out_dir / "manifest.json", manifest)
        raise


def run_lmms_eval_command(args: Any) -> int:
    result = run_lmms_evaluation(args.config)
    print(json.dumps({"out_dir": result["manifest"]["out_dir"], "results": result["results"].get("results", {})}, indent=2))
    return 0
