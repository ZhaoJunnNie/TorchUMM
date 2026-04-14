from __future__ import annotations

import importlib
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
from PIL import Image


def _timeout_handler(signum, frame):
    raise TimeoutError("Image generation timed out")


# Default sampling params matching Emu3.5 configs/config.py
_DEFAULT_SAMPLING_PARAMS = dict(
    use_cache=True,
    text_top_k=1024,
    text_top_p=0.9,
    text_temperature=1.0,
    image_top_k=5120,
    image_top_p=1.0,
    image_temperature=1.0,
    top_k=131072,
    top_p=1.0,
    temperature=1.0,
    num_beams_per_group=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    max_new_tokens=5120,
    guidance_scale=1.0,
    use_differential_sampling=True,
    do_sample=True,
    num_beams=1,
)

_SPECIAL_TOKENS = dict(
    BOS="<|extra_203|>",
    EOS="<|extra_204|>",
    PAD="<|endoftext|>",
    EOL="<|extra_200|>",
    EOF="<|extra_201|>",
    TMS="<|extra_202|>",
    IMG="<|image token|>",
    BOI="<|image start|>",
    EOI="<|image end|>",
    BSS="<|extra_100|>",
    ESS="<|extra_101|>",
    BOG="<|extra_60|>",
    EOG="<|extra_61|>",
    BOC="<|extra_50|>",
    EOC="<|extra_51|>",
)


def _build_unc_and_template(task: str, with_image: bool):
    """Build unconditional prompt and template for Emu3.5."""
    task_str = task.lower()
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>" % task_str
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {question} ASSISTANT: <|extra_100|>" % task_str
    return unc_p, tmpl


class Emu3dot5Backbone:
    name = "emu3_5"

    def __init__(
        self,
        model_path: Optional[str] = None,
        vq_path: Optional[str] = None,
        emu3_5_root: Optional[str] = None,
        use_vllm: bool = True,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.7,
        vq_device: str = "cuda:0",
        classifier_free_guidance: float = 5.0,
        max_new_tokens: int = 5120,
        image_area: int = 1048576,
        seed: int = 6666,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        packaged = Path(__file__).resolve().parent / "Emu3.5"
        self.emu3_5_root = Path(emu3_5_root) if emu3_5_root else (
            packaged if packaged.exists() else repo_root / "model" / "Emu3.5"
        )
        self.model_path = model_path or str(repo_root / "model_cache" / "emu3_5" / "Emu3.5-Image")
        self.vq_path = vq_path or str(repo_root / "model_cache" / "emu3_5" / "Emu3.5-VisionTokenizer")
        self.tokenizer_path = str(self.emu3_5_root / "src" / "tokenizer_emu3_ibq")
        self.use_vllm = use_vllm
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.vq_device = vq_device
        self.classifier_free_guidance = classifier_free_guidance
        self.max_new_tokens = max_new_tokens
        self.image_area = image_area
        self.seed = seed

        # Lazily populated by load()
        self.model: Any = None
        self.tokenizer: Any = None
        self.vq_model: Any = None
        self._backend: str = ""  # "vllm" or "transformers"

    def load(self, cfg: dict[str, Any]) -> None:
        # Update config from dict
        for key in (
            "model_path", "vq_path", "emu3_5_root", "use_vllm",
            "tensor_parallel_size", "gpu_memory_utilization", "vq_device",
            "classifier_free_guidance", "max_new_tokens", "image_area", "seed",
        ):
            if key in cfg and cfg[key] is not None:
                val = cfg[key]
                if key == "emu3_5_root":
                    self.emu3_5_root = Path(val)
                    self.tokenizer_path = str(self.emu3_5_root / "src" / "tokenizer_emu3_ibq")
                else:
                    setattr(self, key, val)

        # Add Emu3.5 repo to sys.path so its modules are importable
        emu_root_str = str(self.emu3_5_root.resolve())
        if emu_root_str not in sys.path:
            sys.path.insert(0, emu_root_str)

        # Build tokenizer (shared by both backends)
        self.tokenizer = self._build_tokenizer()

        # Build VQ vision tokenizer (does NOT depend on modeling_emu3.py)
        vt_mod = importlib.import_module("src.vision_tokenizer")
        self.vq_model = vt_mod.build_vision_tokenizer("ibq", self.vq_path, device=self.vq_device)

        if self.use_vllm:
            try:
                self.model = self._build_vllm_model()
                self._backend = "vllm"
                print("[emu3.5] Model loaded with vLLM backend.", flush=True)
            except Exception as e:
                print(f"[emu3.5] vLLM loading failed: {e}.", flush=True)
                raise

        # Build special token IDs
        self.special_token_ids = {}
        for k, v in _SPECIAL_TOKENS.items():
            self.special_token_ids[k] = self.tokenizer.encode(v)[0]

    def _build_tokenizer(self):
        """Build and configure the Emu3.5 text tokenizer."""
        import os.path as osp
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            special_tokens_file=osp.join(self.tokenizer_path, "emu3_vision_tokens.txt"),
            trust_remote_code=True,
        )
        tokenizer.bos_token = "<|extra_203|>"
        tokenizer.eos_token = "<|extra_204|>"
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eol_token = "<|extra_200|>"
        tokenizer.eof_token = "<|extra_201|>"
        tokenizer.tms_token = "<|extra_202|>"
        tokenizer.img_token = "<|image token|>"
        tokenizer.boi_token = "<|image start|>"
        tokenizer.eoi_token = "<|image end|>"
        tokenizer.bss_token = "<|extra_100|>"
        tokenizer.ess_token = "<|extra_101|>"
        tokenizer.bog_token = "<|extra_60|>"
        tokenizer.eog_token = "<|extra_61|>"
        tokenizer.boc_token = "<|extra_50|>"
        tokenizer.eoc_token = "<|extra_51|>"
        return tokenizer

    def _build_vllm_model(self):
        """Build the vLLM LLM engine.

        Requires BAAI's vLLM patches to be applied at image build time
        (see modal/images.py).  The patches register a native Emu3.5
        architecture in vLLM with optimized attention kernels.
        """
        from vllm import LLM

        print(f"[emu3.5] Loading model with vLLM from {self.model_path} ...", flush=True)

        # Build resolution token map
        resolution_map = {}
        for digit_str in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*"]:
            resolution_map[self.tokenizer.encode(digit_str)[0]] = digit_str

        model = LLM(
            self.model_path,
            tokenizer=self.tokenizer_path,
            trust_remote_code=True,
            dtype="auto",
            tensor_parallel_size=self.tensor_parallel_size,
            distributed_executor_backend="mp",
            gpu_memory_utilization=self.gpu_memory_utilization,
            disable_log_stats=False,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            max_num_batched_tokens=26000,
            max_num_seqs=2,
            seed=self.seed,
            generation_config="vllm",
            scheduler_cls="vllm.v1.core.sched.batch_scheduler.Scheduler",
            compilation_config={
                "full_cuda_graph": True,
                "backend": "cudagraph",
                "cudagraph_capture_sizes": [1, 2],
            },
            additional_config={
                "boi_token_id": self.tokenizer.encode("<|image start|>")[0],
                "soi_token_id": self.tokenizer.encode("<|image token|>")[0],
                "eol_token_id": self.tokenizer.encode("<|extra_200|>")[0],
                "eoi_token_id": self.tokenizer.encode("<|image end|>")[0],
                "resolution_map": resolution_map,
            },
        )
        model.set_tokenizer(self.tokenizer)
        return model

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load({})

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, batch: dict[str, Any], gen_cfg: dict[str, Any]) -> Any:
        self._ensure_loaded()

        prompt = batch.get("prompt") or gen_cfg.get("prompt")
        if prompt is None:
            raise ValueError("Generation requires a prompt.")
        prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
        output_path = batch.get("output_path")

        saved_paths = []
        pil_images: list[Image.Image] = []
        with torch.inference_mode():
            for i, p in enumerate(prompts):
                img = self._generate_one(p, gen_cfg)
                if img is not None:
                    pil_images.append(img)
                    if output_path and i == 0:
                        dst = Path(output_path)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        img.save(str(dst), format=self._fmt(dst))
                        saved_paths.append(str(dst))

        return {
            "images": pil_images if pil_images else saved_paths,
            "saved_paths": saved_paths,
            "output_path": output_path or "",
        }

    def generate_batch(
        self,
        prompt_items: list[dict[str, Any]],
        gen_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not prompt_items:
            return []
        self._ensure_loaded()

        timeout_sec = int(gen_cfg.get("timeout_per_image", 900))

        results: list[dict[str, Any]] = []
        with torch.inference_mode():
            for i, item in enumerate(prompt_items):
                prompt = item["prompt"]
                output_path = item.get("output_path", "")
                print(f"[emu3.5] [{i + 1}/{len(prompt_items)}] {prompt[:80]} ...", flush=True)

                try:
                    prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(timeout_sec)
                    try:
                        img = self._generate_one(prompt, gen_cfg)
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, prev_handler)
                except TimeoutError:
                    print(f"[emu3.5]   Timeout after {timeout_sec}s, skipping", flush=True)
                    results.append({"images": [], "ok": False})
                    continue
                except Exception as e:
                    print(f"[emu3.5]   Error: {e}", flush=True)
                    results.append({"images": [], "ok": False})
                    continue

                ok = False
                if img is not None and output_path:
                    dst = Path(output_path)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    img.save(str(dst), format=self._fmt(dst))
                    ok = dst.is_file()
                    print(f"[emu3.5]   -> saved to {dst}", flush=True)

                results.append({"images": [output_path] if ok else [], "output_path": output_path, "ok": ok})

        return results

    def _generate_one(self, prompt: str, gen_cfg: dict[str, Any]) -> Optional[Image.Image]:
        """Run a single T2I generation and return the PIL Image (or None)."""
        cfg_scale = float(gen_cfg.get("classifier_free_guidance", self.classifier_free_guidance))
        max_tokens = int(gen_cfg.get("max_new_tokens", self.max_new_tokens))
        image_area = int(gen_cfg.get("image_area", self.image_area))

        # Build prompt from template
        unc_prompt, template = _build_unc_and_template("t2i", False)
        full_prompt = template.format(question=prompt)

        # Tokenize
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False)
        bos_id = self.special_token_ids["BOS"]
        if input_ids[0, 0] != bos_id:
            bos = torch.tensor([[bos_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([bos, input_ids], dim=1)

        unconditional_ids = self.tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False)

        # Build a config namespace for the generation functions
        sampling_params = dict(_DEFAULT_SAMPLING_PARAMS)
        sampling_params["max_new_tokens"] = max_tokens
        # Override sampling params from gen_cfg if provided
        for key in ("image_top_k", "text_top_k", "text_top_p", "image_top_p", "do_sample"):
            if key in gen_cfg:
                sampling_params[key] = gen_cfg[key]

        cfg_ns = SimpleNamespace(
            classifier_free_guidance=cfg_scale,
            sampling_params=sampling_params,
            special_token_ids=self.special_token_ids,
            special_tokens=_SPECIAL_TOKENS,
            unconditional_type="no_text",
            image_area=image_area,
            task_type="t2i",
            streaming=False,
        )

        if self._backend == "vllm":
            return self._generate_one_vllm(cfg_ns, input_ids, unconditional_ids)
        else:
            return self._generate_one_transformers(cfg_ns, input_ids, unconditional_ids)

    def _generate_one_vllm(self, cfg_ns, input_ids, unconditional_ids) -> Optional[Image.Image]:
        """Generate one image using the vLLM backend."""
        vllm_gen = importlib.import_module("src.utils.vllm_generation_utils")
        gen_utils = importlib.import_module("src.utils.generation_utils")

        for result_tokens in vllm_gen.generate(cfg_ns, self.model, self.tokenizer, input_ids, unconditional_ids):
            result = self.tokenizer.decode(result_tokens, skip_special_tokens=False)
            mm_out = gen_utils.multimodal_decode(result, self.tokenizer, self.vq_model)
            for kind, payload in mm_out:
                if kind == "image" and isinstance(payload, Image.Image):
                    return payload
        return None

    def _generate_one_transformers(self, cfg_ns, input_ids, unconditional_ids) -> Optional[Image.Image]:
        """Generate one image using the Transformers backend."""
        gen_utils = importlib.import_module("src.utils.generation_utils")

        for result_tokens in gen_utils.generate(cfg_ns, self.model, self.tokenizer, input_ids, unconditional_ids):
            result = self.tokenizer.decode(result_tokens, skip_special_tokens=False)
            mm_out = gen_utils.multimodal_decode(result, self.tokenizer, self.vq_model)
            for kind, payload in mm_out:
                if kind == "image" and isinstance(payload, Image.Image):
                    return payload
        return None

    # ------------------------------------------------------------------
    # Editing (X2I: reference image + prompt → edited image)
    # ------------------------------------------------------------------

    def editing(
        self,
        prompt: Optional[str] = None,
        images: Optional[list[str]] = None,
        output_path: Optional[str] = None,
        editing_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Image editing via Emu3.5's X2I (Any-to-Image) capability.

        Takes one or more reference images and an editing instruction,
        produces the edited image.
        """
        if not prompt:
            raise ValueError("Editing requires a prompt (editing instruction).")
        if not images:
            raise ValueError("Editing requires at least one reference image.")

        self._ensure_loaded()
        editing_cfg = editing_cfg or {}

        cfg_scale = float(editing_cfg.get("classifier_free_guidance", self.classifier_free_guidance))
        max_tokens = int(editing_cfg.get("max_new_tokens", self.max_new_tokens))
        image_area = int(editing_cfg.get("image_area", self.image_area))

        # Import helpers from the Emu3.5 repo
        input_utils = importlib.import_module("src.utils.input_utils")
        img_cfg = SimpleNamespace(image_area=image_area)

        # Emu3.5's x2i mode works best with short, direct prompts.
        # Multi-step pipelines (e.g. uni_mmmu maze/sliding) pass the
        # entire accumulated conversation as the prompt, which causes
        # the model to output text descriptions instead of image tokens.
        # Extract just the last instruction for x2i.
        editing_prompt = self._extract_editing_instruction(prompt)

        # Encode reference images into token strings.
        # Limit to the last few images to keep the input sequence
        # manageable — earlier images are context history, only the
        # most recent state matters for the next generation step.
        _MAX_EDITING_IMAGES = 4
        if len(images) > _MAX_EDITING_IMAGES:
            images = images[-_MAX_EDITING_IMAGES:]
        image_strs: list[str] = []
        for img_path in images:
            pil_image = Image.open(img_path) if isinstance(img_path, str) else img_path
            pil_image = pil_image.convert("RGB")
            image_strs.append(input_utils.build_image(pil_image, img_cfg, self.tokenizer, self.vq_model))
        combined_image_str = "".join(image_strs)

        # Build prompt using x2i template (with_image=True)
        unc_prompt, template = _build_unc_and_template("x2i", True)
        full_prompt = template.format(question=editing_prompt)
        full_prompt = full_prompt.replace("<|IMAGE|>", combined_image_str)
        unc_prompt = unc_prompt.replace("<|IMAGE|>", combined_image_str)

        # Tokenize
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False)
        bos_id = self.special_token_ids["BOS"]
        if input_ids[0, 0] != bos_id:
            bos = torch.tensor([[bos_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([bos, input_ids], dim=1)
        unconditional_ids = self.tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False)

        # Build config namespace
        sampling_params = dict(_DEFAULT_SAMPLING_PARAMS)
        sampling_params["max_new_tokens"] = max_tokens
        for key in ("image_top_k", "text_top_k", "text_top_p", "image_top_p", "do_sample"):
            if key in editing_cfg:
                sampling_params[key] = editing_cfg[key]

        cfg_ns = SimpleNamespace(
            classifier_free_guidance=cfg_scale,
            sampling_params=sampling_params,
            special_token_ids=self.special_token_ids,
            special_tokens=_SPECIAL_TOKENS,
            unconditional_type="no_text",
            image_area=image_area,
            task_type="x2i",
            streaming=False,
        )

        # Generate
        img = None
        if self._backend == "vllm":
            img = self._generate_one_vllm(cfg_ns, input_ids, unconditional_ids)
        else:
            img = self._generate_one_transformers(cfg_ns, input_ids, unconditional_ids)

        saved_paths: list[str] = []
        if img is not None and output_path:
            dst = Path(output_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(dst), format=self._fmt(dst))
            saved_paths.append(str(dst))
            print(f"[emu3.5-editing] saved to {dst}", flush=True)

        return {
            "images": [img] if img else [],
            "image_paths": saved_paths,
            "output_path": output_path or "",
        }

    @staticmethod
    def _extract_editing_instruction(prompt: str) -> str:
        """Extract a concise editing instruction from a potentially long context.

        Multi-step evaluation pipelines (uni_mmmu maze/sliding/jigsaw) pass
        the full accumulated conversation history as the editing prompt.
        Emu3.5's x2i template works best with short instructions — long
        prompts cause the model to generate text instead of image tokens.
        """
        if not prompt:
            return "Generate an image based on the reference."
        lines = [l.strip() for l in prompt.strip().splitlines() if l.strip()]
        # Short prompts (e.g. GEdit instructions): use as-is
        if len(lines) <= 3 and len(prompt) < 500:
            return prompt
        # Long prompts: use the last non-empty line as the instruction
        return lines[-1] if lines else prompt

    # ------------------------------------------------------------------
    # Understanding (VQA / image comprehension)
    # ------------------------------------------------------------------

    _DEFAULT_UNDERSTANDING_CFG: dict[str, Any] = {
        "max_new_tokens": 5120,
        "classifier_free_guidance": 2.0,
        "image_area": 518400,
    }

    def understand(self, batch: dict[str, Any], understanding_cfg: dict[str, Any]) -> Any:
        return self.understanding(
            prompt=batch.get("prompt"),
            images=batch.get("images", []),
            videos=batch.get("videos", []),
            understanding_cfg=understanding_cfg,
        )

    def understanding(
        self,
        prompt: Optional[str] = None,
        images: Optional[list[str]] = None,
        videos: Optional[list[str]] = None,
        understanding_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if videos:
            raise NotImplementedError("Emu3.5 does not support video understanding.")
        if not prompt and not images:
            raise ValueError("At least one of `prompt` or `images` is required.")

        self._ensure_loaded()
        cfg = dict(self._DEFAULT_UNDERSTANDING_CFG)
        if understanding_cfg:
            cfg.update(understanding_cfg)

        # Import helpers from the Emu3.5 repo
        input_utils = importlib.import_module("src.utils.input_utils")

        # Determine whether we have an image
        has_image = bool(images)
        image_str = ""
        if has_image:
            img_path = images[0] if isinstance(images[0], str) else images[0]
            pil_image = Image.open(img_path).convert("RGB")
            # build_image needs a cfg namespace with image_area
            img_cfg = SimpleNamespace(image_area=int(cfg.get("image_area", 518400)))
            image_str = input_utils.build_image(pil_image, img_cfg, self.tokenizer, self.vq_model)

        # Build prompt from template
        unc_prompt, template = _build_unc_and_template("understanding", has_image)
        full_prompt = template.format(question=prompt or "")
        if has_image:
            full_prompt = full_prompt.replace("<|IMAGE|>", image_str)
            unc_prompt = unc_prompt.replace("<|IMAGE|>", image_str)

        # Tokenize
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False)
        bos_id = self.special_token_ids["BOS"]
        if input_ids[0, 0] != bos_id:
            bos = torch.tensor([[bos_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([bos, input_ids], dim=1)
        unconditional_ids = self.tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False)

        # Build config namespace for generation (understanding mode)
        max_tokens = int(cfg.get("max_new_tokens", 5120))
        cfg_scale = float(cfg.get("classifier_free_guidance", 2.0))
        sampling_params = dict(_DEFAULT_SAMPLING_PARAMS)
        sampling_params["max_new_tokens"] = max_tokens
        # Greedy decoding for reproducible evaluation
        sampling_params["do_sample"] = bool(cfg.get("do_sample", False))
        if not sampling_params["do_sample"]:
            sampling_params["temperature"] = 1.0
            sampling_params["text_temperature"] = 1.0
            sampling_params["top_k"] = 1
            sampling_params["text_top_k"] = 1

        cfg_ns = SimpleNamespace(
            classifier_free_guidance=cfg_scale,
            sampling_params=sampling_params,
            special_token_ids=self.special_token_ids,
            special_tokens=_SPECIAL_TOKENS,
            unconditional_type="no_text",
            image_area=int(cfg.get("image_area", 518400)),
            task_type="understanding",  # NOT t2i/x2i → stop at EOS token
            streaming=False,
        )

        # Generate
        vllm_gen = importlib.import_module("src.utils.vllm_generation_utils")
        gen_utils = importlib.import_module("src.utils.generation_utils")

        text_parts: list[str] = []
        with torch.inference_mode():
            for result_tokens in vllm_gen.generate(cfg_ns, self.model, self.tokenizer, input_ids, unconditional_ids):
                result = self.tokenizer.decode(result_tokens, skip_special_tokens=False)
                mm_out = gen_utils.multimodal_decode(result, self.tokenizer, self.vq_model)
                for kind, payload in mm_out:
                    if kind == "text":
                        text_parts.append(payload)

        answer = "".join(text_parts).strip()
        return {"text": answer}

    # ------------------------------------------------------------------
    # Interleaved generation (text + multi-image in one call)
    # ------------------------------------------------------------------

    def generate_interleaved(
        self,
        prompt: str,
        gen_cfg: dict[str, Any],
        images_dir: Optional[Path] = None,
        item_id: str = "",
    ) -> dict[str, Any]:
        """Single-call interleaved generation returning text + multiple images.

        Uses task_type="story" so vLLM stops at EOS (not image_end),
        allowing the model to produce a full interleaved text+image sequence.
        """
        self._ensure_loaded()

        cfg_scale = float(gen_cfg.get("classifier_free_guidance", self.classifier_free_guidance))
        max_tokens = int(gen_cfg.get("max_new_tokens", self.max_new_tokens))
        image_area = int(gen_cfg.get("image_area", self.image_area))

        # Use "story" task for interleaved output (NOT "t2i")
        unc_prompt, template = _build_unc_and_template("story", False)
        full_prompt = template.format(question=prompt)

        # Tokenize (same logic as _generate_one)
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False)
        bos_id = self.special_token_ids["BOS"]
        if input_ids[0, 0] != bos_id:
            bos = torch.tensor([[bos_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([bos, input_ids], dim=1)
        unconditional_ids = self.tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False)

        sampling_params = dict(_DEFAULT_SAMPLING_PARAMS)
        sampling_params["max_new_tokens"] = max_tokens
        # Interleaved mode uses larger image_top_k
        sampling_params["image_top_k"] = int(gen_cfg.get("image_top_k", 10240))
        for key in ("text_top_k", "text_top_p", "image_top_p", "do_sample"):
            if key in gen_cfg:
                sampling_params[key] = gen_cfg[key]

        cfg_ns = SimpleNamespace(
            classifier_free_guidance=cfg_scale,
            sampling_params=sampling_params,
            special_token_ids=self.special_token_ids,
            special_tokens=_SPECIAL_TOKENS,
            unconditional_type="no_text",
            image_area=image_area,
            task_type="story",
            streaming=False,
        )

        # Run generation and collect ALL text + images
        text_parts: list[str] = []
        images: list[Image.Image] = []

        gen_mod = "src.utils.vllm_generation_utils" if self._backend == "vllm" else "src.utils.generation_utils"
        gen_fn = importlib.import_module(gen_mod)
        gen_utils = importlib.import_module("src.utils.generation_utils")

        with torch.inference_mode():
            for result_tokens in gen_fn.generate(cfg_ns, self.model, self.tokenizer, input_ids, unconditional_ids):
                result = self.tokenizer.decode(result_tokens, skip_special_tokens=False)
                mm_out = gen_utils.multimodal_decode(result, self.tokenizer, self.vq_model)
                for kind, payload in mm_out:
                    if kind == "image" and isinstance(payload, Image.Image):
                        images.append(payload)
                    elif kind == "text" and isinstance(payload, str) and payload.strip():
                        text_parts.append(payload.strip())

        # Save images to disk
        saved_paths: list[str] = []
        if images_dir and images:
            images_dir = Path(images_dir)
            images_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(images):
                filename = f"{item_id}_{idx}.png" if item_id else f"image_{idx}.png"
                dst = images_dir / filename
                img.save(str(dst), format="PNG")
                saved_paths.append(str(dst))
                print(f"[emu3.5-interleaved]   -> saved {dst}", flush=True)

        print(
            f"[emu3.5-interleaved] text_len={sum(len(t) for t in text_parts)}, "
            f"images={len(images)}",
            flush=True,
        )
        return {
            "text": " ".join(text_parts),
            "images": images,
            "saved_paths": saved_paths,
        }

    # Official UEval parameters for Emu3.5 (from UEval/generate_outputs/Emu3.5/configs/config.py)
    _UEVAL_GEN_CFG: dict[str, Any] = {
        "classifier_free_guidance": 3.0,
        "max_new_tokens": 32768,
        "image_top_k": 10240,
        "image_area": 518400,
    }

    def run_unified_batch(
        self,
        items: list[dict[str, Any]],
        images_dir: Path,
        understanding_params: dict[str, Any],
        gen_params: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """UEval generation: single interleaved call (story mode) per item.

        Matches the official Emu3.5 UEval implementation — one generation call
        produces both text and images via task_type="story".
        """
        self._ensure_loaded()

        # Merge with official UEval defaults (user overrides take precedence)
        merged_cfg = {**self._UEVAL_GEN_CFG, **gen_params}

        text_results: list[dict[str, Any]] = []
        gen_results: list[dict[str, Any]] = []

        for i, item in enumerate(items):
            prompt = item["prompt_text"]
            item_id = item["item_id"]
            print(f"[emu3.5-unified] [{i + 1}/{len(items)}] id={item_id}", flush=True)

            try:
                result = self.generate_interleaved(
                    prompt=prompt,
                    gen_cfg=merged_cfg,
                    images_dir=images_dir,
                    item_id=item_id,
                )
                text_results.append({"text": result["text"]})
                gen_results.append({"ok": bool(result["saved_paths"])})
            except Exception as e:
                print(f"[emu3.5-unified]   error: {e}", flush=True)
                text_results.append({"text": ""})
                gen_results.append({"ok": False})

        return text_results, gen_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(path: Path) -> str:
        """Infer PIL save format from extension, defaulting to PNG."""
        ext = path.suffix.lower().lstrip(".")
        return {"jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}.get(ext, "PNG")
