from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image


def _conversation_roles(vl_chat_processor: Any) -> tuple[str, str]:
    sft_format = str(getattr(vl_chat_processor, "sft_format", ""))
    if sft_format == "deepseek_old":
        return "User", "Assistant"
    if sft_format == "deepseek":
        return "<|User|>", "<|Assistant|>"
    raise ValueError(
        "Unsupported Janus sft_format "
        f"{sft_format!r}; expected 'deepseek_old' or 'deepseek'"
    )


class JanusProBackbone:
    name = "janus_pro"

    def __init__(
        self,
        model_path: Optional[str] = None,
        janus_root: Optional[str] = None,
        seed: int = 42,
        torch_dtype: str = "bfloat16",
        attn_implementation: Optional[str] = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        self.janus_root = (
            Path(janus_root).expanduser()
            if janus_root
            else repo_root / "src" / "umm" / "backbones" / "janus_pro" / "Janus"
        )
        self.model_path = (
            str(Path(model_path).expanduser())
            if model_path
            else str(repo_root / "model_cache" / "janus" / "models" / "Janus_pro_7B")
        )
        self.seed = seed
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation

        self.default_generation_cfg: dict[str, Any] = {
            "temperature": 1.0,
            "parallel_size": 4,
            "cfg_weight": 5.0,
            "image_token_num_per_image": 576,
            "img_size": 384,
            "patch_size": 16,
        }
        self.default_understanding_cfg: dict[str, Any] = {
            "max_new_tokens": 5120,
            "do_sample": False,
            "use_cache": True,
        }

        self.vl_chat_processor: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.operator_quotient: Any = None
        self.operator_quotient_routes = {
            "understanding": "cayley",
            "generation": "baseline",
        }
        self.operator_quotient_confidence_threshold = 0.0
        self.operator_quotient_ensemble_weight = 0.5
        self.operator_quotient_choice_excluded_labels: set[str] = set()
        self.operator_quotient_report_path: Optional[Path] = None
        self.quantization_mode_router: Any = None
        self.quantization_prefill_selector = False
        self.quantization_prefill_margin_threshold = 0.0

    def load(self, cfg: dict[str, Any]) -> None:
        cfg_model_path = cfg.get("model_path")
        if cfg_model_path:
            self.model_path = str(Path(cfg_model_path).expanduser())
        cfg_janus_root = cfg.get("janus_root")
        if cfg_janus_root:
            self.janus_root = Path(cfg_janus_root).expanduser()
        self.seed = int(cfg.get("seed", self.seed))
        self.torch_dtype = str(cfg.get("torch_dtype", self.torch_dtype))
        self.attn_implementation = cfg.get("attn_implementation", self.attn_implementation)
        generation_cfg = cfg.get("generation_cfg")
        if isinstance(generation_cfg, dict):
            self.default_generation_cfg.update(generation_cfg)
        understanding_cfg = cfg.get("understanding_cfg")
        if isinstance(understanding_cfg, dict):
            self.default_understanding_cfg.update(understanding_cfg)

        self._set_seed(self.seed)
        self.vl_chat_processor, self.tokenizer, self.model = self._build_model()
        operator_quotient_cfg = cfg.get("operator_quotient")
        if operator_quotient_cfg is not None:
            if not isinstance(operator_quotient_cfg, dict):
                raise TypeError("`operator_quotient` must be a mapping when provided")
            self._install_operator_quotient(operator_quotient_cfg)

    def generate(self, batch: dict[str, Any], gen_cfg: dict[str, Any]) -> Any:
        prompt = batch.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Expected batch['prompt'] as a non-empty string.")
        return self.generation(prompt=prompt, output_path=batch.get("output_path"), generation_cfg=gen_cfg)

    def generation(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        generation_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if self.model is None or self.vl_chat_processor is None:
            self.load({})
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Expected `prompt` as a non-empty string.")

        config = dict(self.default_generation_cfg)
        if generation_cfg:
            config.update(generation_cfg)

        self._route_operator_quotient("generation")
        self._route_quantization_mode("generation")
        full_prompt = self._build_generation_prompt(prompt)
        images = self._generate_images(full_prompt, config)
        saved_paths = self._save_images(images, output_path)
        self._write_operator_quotient_report()
        return {
            "image": images[0] if images else None,
            "images": images,
            "saved_paths": saved_paths,
        }

    def understand(self, batch: dict[str, Any], understanding_cfg: dict[str, Any]) -> Any:
        return self.understanding(
            prompt=batch.get("prompt"),
            images=batch.get("images", []),
            videos=batch.get("videos", []),
            metadata=batch.get("metadata"),
            understanding_cfg=understanding_cfg,
        )

    def understanding(
        self,
        prompt: Optional[str] = None,
        images: Optional[list[str]] = None,
        videos: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        understanding_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if self.model is None or self.vl_chat_processor is None:
            self.load({})

        image_paths = images or []
        video_paths = videos or []
        if video_paths:
            raise NotImplementedError("JanusProBackbone.understanding currently does not support videos.")

        if not prompt and not image_paths:
            raise ValueError("Understanding requires at least one input: prompt or images.")

        pil_images: list[Image.Image] = []
        if image_paths:
            image_path = Path(image_paths[0]).expanduser()
            if not image_path.exists():
                raise FileNotFoundError(f"Understanding image not found: {image_path}")
            pil_images.append(Image.open(image_path).convert("RGB"))

        user_prompt = prompt if isinstance(prompt, str) and prompt else "Describe the image."
        content = f"<image_placeholder>\n{user_prompt}" if pil_images else user_prompt
        user_role, assistant_role = _conversation_roles(self.vl_chat_processor)
        conversation = [
            {
                "role": user_role,
                "content": content,
                "images": image_paths[:1] if image_paths else [],
            },
            {"role": assistant_role, "content": ""},
        ]

        self._route_operator_quotient("understanding")
        self._route_quantization_mode("understanding")
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        ).to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        config = dict(self.default_understanding_cfg)
        if understanding_cfg:
            config.update(understanding_cfg)

        quantization_route_diagnostics = None
        if self.quantization_prefill_selector:
            _, quantization_route_diagnostics = (
                self._select_quantization_confidence_route(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    metadata=metadata,
                )
            )

        route = self.operator_quotient_routes["understanding"]
        route_diagnostics = None
        if route == "confidence_cayley_prefill_decode_baseline":
            route, route_diagnostics = self._select_confidence_route(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                metadata=metadata,
            )
        if route == "unanimous_cayley_choice_fallback_baseline":
            selected_choice, route_diagnostics = self._select_unanimous_cayley_choice(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                metadata=metadata,
            )
            if selected_choice is not None:
                answer = selected_choice
            else:
                outputs = self._generate_understanding_tokens(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    config=config,
                    route="baseline",
                )
                answer = self.tokenizer.decode(
                    outputs[0].cpu().tolist(), skip_special_tokens=True
                )
        else:
            outputs = self._generate_understanding_tokens(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                config=config,
                route=route,
            )
            answer = self.tokenizer.decode(
                outputs[0].cpu().tolist(), skip_special_tokens=True
            )
        self._write_operator_quotient_report()
        return {
            "text": answer,
            "sft_format": prepare_inputs["sft_format"][0],
            "operator_route": route,
            "operator_route_diagnostics": route_diagnostics,
            "quantization_route_diagnostics": quantization_route_diagnostics,
        }


    def _install_operator_quotient(self, cfg: dict[str, Any]) -> None:
        if self.model is None:
            raise RuntimeError("Janus model must be loaded before installing operator quotient")
        project_root_value = cfg.get("project_root")
        if not isinstance(project_root_value, str) or not project_root_value:
            raise ValueError("operator_quotient.project_root is required")
        project_root = Path(project_root_value).expanduser().resolve()
        if not project_root.is_dir():
            raise FileNotFoundError(f"operator quotient project root not found: {project_root}")
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)

        deployment_module = importlib.import_module(
            "operator_equivalence.janus_cayley_deployment"
        )
        understanding_route = str(cfg.get("understanding_route", "cayley"))
        generation_route = str(cfg.get("generation_route", "baseline"))
        understanding_routes = {
            "baseline",
            "cayley",
            "prefill_cayley_decode_baseline",
            "prefill_logit_ensemble_decode_baseline",
            "confidence_cayley_prefill_decode_baseline",
            "unanimous_cayley_choice_fallback_baseline",
        }
        if understanding_route not in understanding_routes:
            raise ValueError(
                "operator quotient understanding route must be baseline, cayley, "
                "prefill_cayley_decode_baseline, "
                "prefill_logit_ensemble_decode_baseline, or "
                "confidence_cayley_prefill_decode_baseline, or "
                "unanimous_cayley_choice_fallback_baseline"
            )
        if generation_route not in {"baseline", "cayley"}:
            raise ValueError("operator quotient generation route must be baseline or cayley")
        self.operator_quotient_routes = {
            "understanding": understanding_route,
            "generation": generation_route,
        }
        self.operator_quotient_confidence_threshold = float(
            cfg.get("confidence_delta_margin_threshold", 0.0)
        )
        self.operator_quotient_ensemble_weight = float(
            cfg.get("ensemble_cayley_weight", 0.5)
        )
        if not 0.0 <= self.operator_quotient_ensemble_weight <= 1.0:
            raise ValueError("operator quotient ensemble_cayley_weight must be in [0, 1]")
        excluded_labels = cfg.get("unanimous_choice_excluded_labels", [])
        if not isinstance(excluded_labels, list):
            raise TypeError(
                "operator quotient unanimous_choice_excluded_labels must be a list"
            )
        self.operator_quotient_choice_excluded_labels = {
            str(label).strip().upper() for label in excluded_labels
        }
        invalid_labels = self.operator_quotient_choice_excluded_labels.difference(
            "ABCDEFGHIJKLM"
        )
        if invalid_labels:
            raise ValueError(
                "operator quotient unanimous_choice_excluded_labels contains "
                f"unsupported labels: {sorted(invalid_labels)}"
            )
        report_path = cfg.get("report_path")
        if report_path:
            self.operator_quotient_report_path = Path(str(report_path)).expanduser().resolve()

        self.operator_quotient = deployment_module.install_mode_routed_cayley_w4a8(
            self.model,
            correction_map_path=cfg["correction_map_path"],
            activation_result_path=cfg["activation_result_path"],
            cayley_calibration_result_path=cfg["cayley_calibration_result_path"],
            initial_state=self.operator_quotient_routes["generation"],
            weight_row_chunk=int(cfg.get("weight_row_chunk", 256)),
            cayley_strength=float(cfg.get("cayley_strength", 1.0)),
            source_paths=cfg.get("source_paths"),
        )
        self._write_operator_quotient_report()

    def _route_operator_quotient(self, mode: str) -> None:
        if self.operator_quotient is None:
            return
        route = self.operator_quotient_routes[mode]
        if route in {
            "prefill_cayley_decode_baseline",
            "confidence_cayley_prefill_decode_baseline",
        }:
            route = "cayley"
        elif route in {
            "prefill_logit_ensemble_decode_baseline",
            "unanimous_cayley_choice_fallback_baseline",
        }:
            route = "baseline"
        self.operator_quotient.install(route)

    def _route_quantization_mode(self, mode: str) -> None:
        if self.quantization_mode_router is not None:
            self.quantization_mode_router(mode)

    def _select_unanimous_cayley_choice(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Optional[dict[str, Any]],
    ) -> tuple[Optional[str], dict[str, Any]]:
        choices_raw = metadata.get("choices") if isinstance(metadata, dict) else None
        if not isinstance(choices_raw, dict) or len(choices_raw) < 2:
            return None, {
                "reason": "non_multiple_choice_or_missing_choices",
                "selected": "baseline_generation",
            }
        choices = [choice for choice in "ABCDEFGHIJKLM" if choice in choices_raw]
        if len(choices) < 2:
            return None, {
                "reason": "unsupported_choice_labels",
                "selected": "baseline_generation",
            }

        stems = {
            "empty": ("", True),
            "answer_colon": ("Answer:", True),
            "answer_is": ("The answer is", True),
            "answer_paren": ("The answer is (", False),
        }
        language_model = self.model.language_model
        self.operator_quotient.install("cayley")
        predictions: dict[str, str] = {}
        scores: dict[str, dict[str, float]] = {}
        with torch.inference_mode():
            for name, (stem, leading_space) in stems.items():
                stem_ids = self.tokenizer.encode(stem, add_special_tokens=False)
                stem_embeds = (
                    language_model.get_input_embeddings()(
                        torch.tensor(
                            stem_ids,
                            dtype=torch.long,
                            device=inputs_embeds.device,
                        ).unsqueeze(0)
                    )
                    if stem_ids
                    else inputs_embeds[:, :0]
                )
                stem_mask = torch.ones(
                    (1, len(stem_ids)),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                prefix_embeds = torch.cat([inputs_embeds, stem_embeds], dim=1)
                prefix_mask = torch.cat([attention_mask, stem_mask], dim=1)
                candidates = [
                    f" {choice}" if leading_space else choice for choice in choices
                ]
                totals = self._choice_candidate_scores(
                    prefix_embeds=prefix_embeds,
                    prefix_mask=prefix_mask,
                    candidates=candidates,
                )
                best_index = max(range(len(totals)), key=totals.__getitem__)
                predictions[name] = choices[best_index]
                scores[name] = {
                    choice: totals[index] for index, choice in enumerate(choices)
                }
        self.operator_quotient.install("baseline")
        unanimous = len(set(predictions.values())) == 1
        unanimous_choice = next(iter(predictions.values())) if unanimous else None
        excluded = unanimous_choice in self.operator_quotient_choice_excluded_labels
        selected_choice = unanimous_choice if unanimous and not excluded else None
        return selected_choice, {
            "reason": (
                "excluded_unanimous_label_prior"
                if excluded
                else "four_prompt_unanimous_cayley_choice"
                if unanimous
                else "cayley_choice_disagreement"
            ),
            "predictions": predictions,
            "scores": scores,
            "excluded_labels": sorted(
                self.operator_quotient_choice_excluded_labels
            ),
            "selected": selected_choice or "baseline_generation",
        }

    def _choice_candidate_scores(
        self,
        *,
        prefix_embeds: torch.Tensor,
        prefix_mask: torch.Tensor,
        candidates: list[str],
    ) -> list[float]:
        language_model = self.model.language_model
        token_rows = [
            self.tokenizer.encode(text, add_special_tokens=False)
            for text in candidates
        ]
        if any(not row for row in token_rows):
            raise RuntimeError("choice candidate tokenization produced an empty sequence")
        max_tokens = max(len(row) for row in token_rows)
        token_ids = torch.full(
            (len(token_rows), max_tokens),
            int(self.tokenizer.eos_token_id),
            dtype=torch.long,
            device=prefix_embeds.device,
        )
        candidate_mask = torch.zeros_like(token_ids)
        for index, row in enumerate(token_rows):
            token_ids[index, : len(row)] = torch.tensor(
                row, dtype=torch.long, device=token_ids.device
            )
            candidate_mask[index, : len(row)] = 1

        candidate_embeds = language_model.get_input_embeddings()(token_ids)
        batch = len(token_rows)
        combined_embeds = torch.cat(
            [prefix_embeds.expand(batch, -1, -1), candidate_embeds], dim=1
        )
        combined_mask = torch.cat(
            [prefix_mask.expand(batch, -1), candidate_mask], dim=1
        )
        prefix_length = int(prefix_embeds.shape[1])
        hidden = language_model.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        prediction_hidden = hidden[
            :, prefix_length - 1 : prefix_length - 1 + max_tokens
        ]
        log_probs = language_model.lm_head(prediction_hidden).float().log_softmax(dim=-1)
        totals: list[float] = []
        for index, row in enumerate(token_rows):
            length = len(row)
            targets = token_ids[index, :length]
            values = log_probs[index, :length].gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            totals.append(float(values.sum().item()))
        return totals

    def _select_confidence_route(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Optional[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        choices_raw = metadata.get("choices") if isinstance(metadata, dict) else None
        if not isinstance(choices_raw, dict) or len(choices_raw) < 2:
            return "baseline", {
                "reason": "non_multiple_choice_or_missing_choices",
                "selected": "baseline",
            }
        choices = [choice for choice in "ABCDEFGHIJKLM" if choice in choices_raw]
        if len(choices) < 2:
            return "baseline", {
                "reason": "unsupported_choice_labels",
                "selected": "baseline",
            }

        variant_ids = {
            choice: tuple(
                dict.fromkeys(
                    self.tokenizer.encode(text, add_special_tokens=False)[0]
                    for text in (choice, f" {choice}")
                )
            )
            for choice in choices
        }
        margins: dict[str, float] = {}
        probabilities: dict[str, dict[str, float]] = {}
        with torch.inference_mode():
            for state in ("baseline", "cayley"):
                self.operator_quotient.install(state)
                output = self.model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                log_probs = output.logits[0, -1].float().log_softmax(dim=-1)
                option_scores = torch.stack(
                    [
                        torch.logsumexp(log_probs[list(variant_ids[choice])], dim=0)
                        for choice in choices
                    ]
                )
                conditional = option_scores.softmax(dim=0)
                ordered = torch.sort(conditional, descending=True).values
                margins[state] = float((ordered[0] - ordered[1]).item())
                probabilities[state] = {
                    choice: float(conditional[index].item())
                    for index, choice in enumerate(choices)
                }
        delta_margin = margins["cayley"] - margins["baseline"]
        use_cayley = delta_margin < self.operator_quotient_confidence_threshold
        selected = (
            "prefill_cayley_decode_baseline" if use_cayley else "baseline"
        )
        return selected, {
            "reason": "frozen_delta_margin_threshold",
            "threshold": self.operator_quotient_confidence_threshold,
            "delta_margin": delta_margin,
            "margins": margins,
            "option_probabilities": probabilities,
            "selected": selected,
        }

    def _select_quantization_confidence_route(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Optional[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Select a quantized fiber only when its option decision self-certifies.

        The candidate noninvertible section is rejected unless it and the
        unchanged QuaRot arm choose the same option and its conditional top-two
        option margin improves by more than the frozen threshold. Inputs without
        structured multiple-choice metadata always remain on QuaRot.
        """
        choices_raw = metadata.get("choices") if isinstance(metadata, dict) else None
        if not isinstance(choices_raw, dict) or len(choices_raw) < 2:
            selected = "understanding_static"
            self._route_quantization_mode(selected)
            return selected, {
                "reason": "non_multiple_choice_or_missing_choices",
                "selected": selected,
            }
        choices = [choice for choice in "ABCDEFGHIJKLM" if choice in choices_raw]
        if len(choices) < 2:
            selected = "understanding_static"
            self._route_quantization_mode(selected)
            return selected, {
                "reason": "unsupported_choice_labels",
                "selected": selected,
            }

        variant_ids: dict[str, tuple[int, ...]] = {}
        for choice in choices:
            ids = []
            for text in (choice, f" {choice}"):
                encoded = self.tokenizer.encode(text, add_special_tokens=False)
                if not encoded:
                    raise RuntimeError(
                        f"choice label {choice!r} produced empty tokenization"
                    )
                ids.append(int(encoded[0]))
            variant_ids[choice] = tuple(dict.fromkeys(ids))

        margins: dict[str, float] = {}
        predictions: dict[str, str] = {}
        probabilities: dict[str, dict[str, float]] = {}
        arm_modes = {
            "static": "understanding_static",
            "candidate": "understanding_candidate",
        }
        with torch.inference_mode():
            for arm, mode in arm_modes.items():
                self._route_quantization_mode(mode)
                output = self.model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                log_probs = output.logits[0, -1].float().log_softmax(dim=-1)
                option_scores = torch.stack(
                    [
                        torch.logsumexp(log_probs[list(variant_ids[choice])], dim=0)
                        for choice in choices
                    ]
                )
                conditional = option_scores.softmax(dim=0)
                order = torch.argsort(conditional, descending=True)
                predictions[arm] = choices[int(order[0].item())]
                margins[arm] = float(
                    (conditional[order[0]] - conditional[order[1]]).item()
                )
                probabilities[arm] = {
                    choice: float(conditional[index].item())
                    for index, choice in enumerate(choices)
                }

        delta_margin = margins["candidate"] - margins["static"]
        agrees = predictions["candidate"] == predictions["static"]
        threshold = float(self.quantization_prefill_margin_threshold)
        use_candidate = agrees and delta_margin > threshold
        selected = arm_modes["candidate" if use_candidate else "static"]
        self._route_quantization_mode(selected)
        return selected, {
            "reason": (
                "agreement_and_margin_improvement"
                if use_candidate
                else "arm_disagreement"
                if not agrees
                else "insufficient_margin_improvement"
            ),
            "threshold": threshold,
            "delta_margin": delta_margin,
            "margins": margins,
            "predictions": predictions,
            "option_probabilities": probabilities,
            "selected": selected,
        }

    def _generate_understanding_tokens(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        config: dict[str, Any],
        route: str,
    ) -> torch.Tensor:
        language_model = self.model.language_model
        generation_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "pad_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": int(config.get("max_new_tokens", 512)),
            "do_sample": bool(config.get("do_sample", False)),
            "use_cache": bool(config.get("use_cache", True)),
        }
        phase_routed = {
            "prefill_cayley_decode_baseline",
            "prefill_logit_ensemble_decode_baseline",
        }
        if self.operator_quotient is None or route not in phase_routed:
            if self.operator_quotient is not None:
                self.operator_quotient.install(route)
            return language_model.generate(**generation_kwargs)

        original_forward = language_model.forward

        def phase_routed_forward(*args: Any, **kwargs: Any) -> Any:
            cache = kwargs.get("past_key_values")
            has_cache = False
            if cache is not None:
                get_seq_length = getattr(cache, "get_seq_length", None)
                if callable(get_seq_length):
                    has_cache = int(get_seq_length()) > 0
                else:
                    try:
                        if len(cache) > 0:
                            first_key = cache[0][0]
                            has_cache = int(first_key.shape[-2]) > 0
                    except (IndexError, TypeError, AttributeError):
                        has_cache = True
            if has_cache:
                self.operator_quotient.install("baseline")
                return original_forward(*args, **kwargs)
            if route == "prefill_cayley_decode_baseline":
                self.operator_quotient.install("cayley")
                return original_forward(*args, **kwargs)

            candidate_kwargs = dict(kwargs)
            candidate_kwargs["use_cache"] = False
            candidate_kwargs.pop("past_key_values", None)
            self.operator_quotient.install("cayley")
            candidate_output = original_forward(*args, **candidate_kwargs)
            self.operator_quotient.install("baseline")
            baseline_output = original_forward(*args, **kwargs)
            weight = self.operator_quotient_ensemble_weight
            baseline_output.logits = torch.lerp(
                baseline_output.logits.float(),
                candidate_output.logits.float(),
                weight,
            ).to(dtype=baseline_output.logits.dtype)
            self.operator_quotient.install("baseline")
            return baseline_output

        language_model.forward = phase_routed_forward
        try:
            return language_model.generate(**generation_kwargs)
        finally:
            language_model.forward = original_forward

    def _write_operator_quotient_report(self) -> None:
        if self.operator_quotient is None or self.operator_quotient_report_path is None:
            return
        self.operator_quotient_report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.operator_quotient_report_path.with_suffix(
            self.operator_quotient_report_path.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(self.operator_quotient.report(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.operator_quotient_report_path)


    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_model(self) -> tuple[Any, Any, Any]:
        if not self.janus_root.exists():
            raise FileNotFoundError(f"Janus root not found: {self.janus_root}")

        janus_root_str = str(self.janus_root.resolve())
        if janus_root_str not in sys.path:
            sys.path.insert(0, janus_root_str)

        if not torch.cuda.is_available():
            raise RuntimeError("JanusProBackbone requires at least one CUDA device.")

        transformers_module = importlib.import_module("transformers")
        janus_models = importlib.import_module("janus.models")

        model_path = str(Path(self.model_path).expanduser())
        vl_chat_processor = janus_models.VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        # Force CPU as default device during model loading to prevent
        # transformers 5.x from using meta tensors internally, which
        # breaks torch.linspace(...).item() in siglip_vit.py.
        default_device_before = None
        if hasattr(torch, "set_default_device"):
            if hasattr(torch, "get_default_device"):
                default_device_before = torch.get_default_device()
            torch.set_default_device("cpu")

        try:
            load_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": False,
                "device_map": None,
            }
            if self.attn_implementation:
                config = transformers_module.AutoConfig.from_pretrained(
                    model_path, trust_remote_code=True
                )
                language_config = getattr(config, "language_config", None)
                if language_config is not None:
                    language_config._attn_implementation = self.attn_implementation
                    language_config.attn_implementation = self.attn_implementation
                load_kwargs["config"] = config
            model = transformers_module.AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs,
            )
        finally:
            if hasattr(torch, "set_default_device"):
                torch.set_default_device(default_device_before)

        model = model.to(self._resolve_torch_dtype(self.torch_dtype)).cuda().eval()
        return vl_chat_processor, tokenizer, model

    def _build_generation_prompt(self, prompt: str) -> str:
        user_role, assistant_role = _conversation_roles(self.vl_chat_processor)
        conversation = [
            {"role": user_role, "content": prompt},
            {"role": assistant_role, "content": ""},
        ]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        return sft_format + self.vl_chat_processor.image_start_tag

    def _generate_images(self, prompt: str, cfg: dict[str, Any]) -> list[Image.Image]:
        temperature = float(cfg.get("temperature", 1.0))
        parallel_size = int(cfg.get("parallel_size", 4))
        cfg_weight = float(cfg.get("cfg_weight", 5.0))
        image_token_num_per_image = int(cfg.get("image_token_num_per_image", 576))
        img_size = int(cfg.get("img_size", 384))
        patch_size = int(cfg.get("patch_size", 16))

        input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).cuda()

        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        with torch.inference_mode():
            for i in range(image_token_num_per_image):
                outputs = self.model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=outputs.past_key_values if i != 0 else None,
                )
                hidden_states = outputs.last_hidden_state
                logits = self.model.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(logits / temperature, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)
                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = self.model.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)

            decoded = self.model.gen_vision_model.decode_code(
                generated_tokens.to(dtype=torch.int),
                shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
            )
            decoded = decoded.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)

        return [Image.fromarray(decoded[i]) for i in range(parallel_size)]

    @staticmethod
    def _fmt(path: Path) -> str:
        """Infer PIL save format from extension, defaulting to PNG."""
        ext = path.suffix.lower().lstrip(".")
        return {"jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}.get(ext, "PNG")

    def _save_images(self, images: list[Image.Image], output_path: Optional[str]) -> list[str]:
        if not output_path:
            return []

        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix if out_path.suffix else ".png"
        stem = out_path.stem if out_path.stem else "output"

        saved_paths: list[str] = []
        if len(images) == 1:
            target = out_path if out_path.suffix else out_path.with_suffix(suffix)
            images[0].save(target, format=self._fmt(target))
            saved_paths.append(str(target))
            return saved_paths

        for idx, image in enumerate(images):
            target = out_path.with_name(f"{stem}_{idx}{suffix}")
            image.save(target, format=self._fmt(target))
            saved_paths.append(str(target))
        return saved_paths

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
        lowered = dtype_name.lower()
        if lowered in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if lowered in {"fp16", "float16", "half"}:
            return torch.float16
        if lowered in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
