from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ``${VAR}`` patterns in config string values."""
    if isinstance(obj, str):
        return _ENV_VAR_RE.sub(
            lambda m: os.environ.get(m.group(1), m.group(0)), obj
        )
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. Install with `pip install pyyaml`."
            ) from exc
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return _expand_env_vars(data) if isinstance(data, dict) else {}

    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return _expand_env_vars(data) if isinstance(data, dict) else {}

    raise ValueError(f"Unsupported config format: {p}")
