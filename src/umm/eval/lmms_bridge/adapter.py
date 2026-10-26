"""Bridge lmms-eval generation requests to one already loaded UMM pipeline."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from lmms_eval.api.model import lmms


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


class UMMGenerationAdapter(lmms):
    """Single-process, single-image, greedy adapter with unchanged task scoring.

    The caller owns ``pipeline`` and its loaded model. Construction and cleanup
    never load, replace, quantize, or unload that object. All task prompts are
    passed through unchanged; raw text is saved before stop-string truncation.
    """

    is_simple = True

    def __init__(
        self,
        *,
        pipeline: Any,
        raw_output_path: str | Path,
        image_dir: str | Path,
        generation_defaults: Mapping[str, Any] | None = None,
        generation_overrides: Mapping[str, Any] | None = None,
        model_identity: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not callable(getattr(pipeline, "run", None)):
            raise TypeError("pipeline must expose run(payload).")
        if not isinstance(getattr(pipeline, "backbone_name", None), str):
            raise TypeError("pipeline must expose its backbone_name.")
        backbone = getattr(pipeline, "backbone", None)
        routes = getattr(backbone, "operator_quotient_routes", {})
        if getattr(backbone, "quantization_prefill_selector", False) or (
            isinstance(routes, Mapping)
            and routes.get("understanding") in {
                "confidence_cayley_prefill_decode_baseline",
                "unanimous_cayley_choice_fallback_baseline",
            }
        ):
            raise NotImplementedError("This bridge supports static pipelines; choice-dependent dynamic routing needs its own metadata contract.")
        self.pipeline = pipeline
        self.raw_output_path = Path(raw_output_path)
        self.image_dir = Path(image_dir)
        self.generation_defaults = copy.deepcopy(dict(generation_defaults or {}))
        self.generation_overrides = copy.deepcopy(dict(generation_overrides or {}))
        # Validate provenance before inference, and normalize common Path/NumPy
        # values so a successful answer cannot be lost to late JSON failures.
        self.model_identity = json.loads(json.dumps(dict(model_identity or {}), default=_json_value))
        self.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        # Reusing a response file could silently mix quantized arms or attempts.
        with self.raw_output_path.open("x", encoding="utf-8"):
            pass
        self._request_index = 0

    @property
    def batch_size(self) -> int:
        return 1

    def loglikelihood(self, requests):
        raise NotImplementedError("The UMM bridge implements generate_until, not loglikelihood/PPL.")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("The UMM bridge currently supports single-turn generation.")

    def clean(self) -> None:
        """Keep the caller-owned pipeline and quantized model alive."""

    def _generation_config(self, task_kwargs: Mapping[str, Any]):
        if not isinstance(task_kwargs, Mapping):
            raise TypeError("lmms generation kwargs must be a mapping.")
        effective = {"max_new_tokens": 128, "do_sample": False}
        effective.update(copy.deepcopy(self.generation_defaults))
        effective.update(copy.deepcopy(dict(task_kwargs)))
        effective.update(copy.deepcopy(self.generation_overrides))
        allowed = {
            "max_new_tokens", "do_sample", "use_cache", "until",
            "temperature", "top_p", "top_k", "num_beams", "num_return_sequences",
            "image_aspect_ratio",
        }
        unknown = set(effective) - allowed
        if unknown:
            raise ValueError(f"Unsupported UMM generation parameters: {sorted(unknown)}")
        budget = effective["max_new_tokens"]
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("max_new_tokens must be a positive integer.")
        if effective["do_sample"] is not False:
            raise NotImplementedError("The initial UMM lmms bridge supports greedy decoding (do_sample=false).")
        # Existing Janus/Emu3.5 adapters do not forward these sampling controls.
        # Accept only neutral task defaults, rather than pretend they take effect.
        neutral = {
            "temperature": (0, 0.0, 1, 1.0, None),
            "top_p": (1, 1.0, None),
            "top_k": (0, -1, None),
            "num_beams": (1,),
            "num_return_sequences": (1,),
            "image_aspect_ratio": ("original", None),
        }
        for key, values in neutral.items():
            if key in effective and effective[key] not in values:
                raise NotImplementedError(f"Existing UMM adapters cannot apply {key}={effective[key]!r}.")
        stops = effective.get("until", [])
        if stops is None:
            stops = []
        if isinstance(stops, str):
            stops = [stops]
        if not isinstance(stops, (list, tuple)) or any(not isinstance(stop, str) for stop in stops):
            raise ValueError("until must be a string or a sequence of strings.")
        stops = [stop for stop in stops if stop]
        params = {"max_new_tokens": budget, "do_sample": False}
        if "use_cache" in effective:
            if not isinstance(effective["use_cache"], bool):
                raise ValueError("use_cache must be a boolean.")
            if self.pipeline.backbone_name.startswith("emu3"):
                if not effective["use_cache"]:
                    raise NotImplementedError("The Emu3.5 adapter does not expose disabling its inference cache.")
            else:
                params["use_cache"] = effective["use_cache"]
        return params, stops, effective

    @staticmethod
    def _text(output: Any) -> str:
        if isinstance(output, str):
            return output
        if isinstance(output, Mapping):
            for key in ("text", "response", "answer", "generated_text"):
                if key in output and isinstance(output[key], str):
                    return output[key]
        raise TypeError("UMM understanding output must contain an explicit text string.")

    def _images(self, visuals: Any, request_index: int):
        if visuals is None:
            visuals = []
        if not isinstance(visuals, (list, tuple)):
            raise TypeError("doc_to_visual must return a list of PIL images or image paths.")
        if len(visuals) > 1:
            raise NotImplementedError("The current Janus/Emu3.5 UMM adapters accept one image per request.")
        paths = []
        image_metadata = []
        for index, visual in enumerate(visuals):
            if isinstance(visual, Image.Image):
                image = visual.convert("RGB")
            elif isinstance(visual, (str, Path)):
                with Image.open(visual) as opened:
                    image = opened.convert("RGB")
            else:
                raise TypeError(f"Unsupported visual input: {type(visual).__name__}")
            path = self.image_dir / f"{request_index:08d}_{index}.png"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite bridged image: {path}")
            image.save(path, format="PNG")
            paths.append(str(path))
            image_metadata.append({
                "path": str(path), "width": image.width, "height": image.height,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        return paths, image_metadata

    def generate_until(self, requests) -> list[str]:
        answers: list[str] = []
        for request in requests:
            args = request.args
            if len(args) != 6:
                raise ValueError("Expected lmms request (context, kwargs, visual_fn, doc_id, task, split).")
            context, task_kwargs, visual_fn, doc_id, task, split = args
            if not isinstance(context, str):
                raise TypeError("The task prompt must be text.")
            params, stops, effective = self._generation_config(task_kwargs)
            # Missing IDs or malformed task data must propagate as real errors.
            document = self.task_dict[task][split][doc_id]
            visuals = visual_fn(document)
            request_index = self._request_index
            self._request_index += 1
            paths, image_metadata = self._images(visuals, request_index)
            payload = {
                "backbone": self.pipeline.backbone_name,
                "task": "understanding",
                "prompt": context,
                "images": paths,
                "params": params,
                "metadata": {"lmms_task": task, "lmms_split": split, "lmms_doc_id": doc_id},
            }
            output = self.pipeline.run(payload)
            raw_response = self._text(output)
            positions = [raw_response.find(stop) for stop in stops if stop in raw_response]
            response = raw_response[:min(positions)] if positions else raw_response
            record = {
                "request_index": request_index, "task": task, "split": split, "doc_id": doc_id,
                "prompt": context, "images": image_metadata,
                "upstream_generation_kwargs": dict(task_kwargs),
                "effective_generation_kwargs": effective,
                "generation_kwargs": params, "until": stops,
                "raw_response": raw_response, "response": response,
                "model_identity": self.model_identity,
            }
            if isinstance(output, Mapping):
                record["pipeline_diagnostics"] = {
                    key: output[key] for key in (
                        "operator_route", "operator_route_diagnostics",
                        "route_diagnostics", "quantization_route_diagnostics"
                    ) if key in output
                }
            with self.raw_output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=_json_value) + "\n")
            answers.append(response)
        return answers
