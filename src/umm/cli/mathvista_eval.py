from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

from tqdm import tqdm

from umm.core.config import load_config


DS_COLLECTIONS = {
    "MathVista_testmini": {
        "root": "AI4Math/MathVista",
        "split": "testmini",
        "max_new_tokens": 4096,
    },
    "MathVista_test": {
        "root": "AI4Math/MathVista",
        "split": "test",
        "max_new_tokens": 4096,
    },
}


COT_INSTRUCTION = (
    "Your task is to answer the question below. "
    "Give step by step reasoning before you answer, and when you're ready to answer, "
    "please use the format \"Final answer: ..\""
    "\n\n"
    "Question:"
    "\n\n"
    "{question}"
)


# Load few-shot demo prompt from the canonical source
from umm.eval.internvl_chat.eval.mathvista.prompts.ext_ans import demo_prompt as _EXTRACT_DEMO_PROMPT


# A capable local extraction model can answer MathVista questions itself when
# the target question is included in the extraction prompt.  That turns
# nonsensical model responses into apparently correct predictions.  Keep the
# extractor response-grounded and make invalid extractions explicit so the
# downstream scorer cannot map them to the edit-distance-nearest choice.
EXTRACTION_PROTOCOL = "response_grounded_v1"
INVALID_EXTRACTION = "__INVALID_RESPONSE_EXTRACTION__"


def _normalized_surface(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _surface_contains(response: str, answer: str) -> bool:
    """Return whether an answer occurs as a bounded surface in a response."""

    haystack = _normalized_surface(response)
    needle = _normalized_surface(answer)
    if not haystack or not needle:
        return False
    # Treat thousands separators as formatting, not semantic content.
    if re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", needle):
        return bool(re.search(
            rf"(?<![\w.]){re.escape(needle.replace(',', ''))}(?![\w.])",
            haystack.replace(",", ""),
        ))
    return bool(re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack))


def _ground_extraction(
    response: str,
    extraction: Any,
    problem: dict[str, Any],
) -> str | None:
    """Canonicalize an extraction only when the response supports it.

    The extraction model is an information extractor, not a second benchmark
    solver.  Multiple-choice answers must be an explicit option label or an
    option surface present in the response.  Free-form answers must occur in
    the response; list answers may be reformatted but every scalar element
    must be present.
    """

    raw = str(extraction or "").strip()
    response_text = str(response or "")
    if not raw or raw == INVALID_EXTRACTION or not response_text.strip():
        return None

    if str(problem.get("question_type", "")) == "multi_choice":
        choices = [str(choice) for choice in (problem.get("choices") or [])]
        options = [chr(ord("A") + index) for index in range(len(choices))]

        letter_match = re.fullmatch(r"\s*\(?([A-Za-z])\)?[.)]?\s*", raw)
        if letter_match:
            letter = letter_match.group(1).upper()
            if letter not in options:
                return None
            explicit = re.search(
                rf"(?:final\s+answer|correct\s+answer|answer|option|choice)"
                rf"\s*(?:is\s*)?[:=-]?\s*\(?{letter}\)?(?:\W|$)",
                response_text,
                flags=re.IGNORECASE,
            )
            parenthesized = re.search(
                rf"(?<!\w)\({letter}\)(?!\w)", response_text, flags=re.IGNORECASE
            )
            terminal = re.search(
                rf"(?:^|\n)\s*\(?{letter}\)?[.)]?\s*$",
                response_text,
                flags=re.IGNORECASE,
            )
            choice_surface = _surface_contains(
                response_text, choices[options.index(letter)]
            )
            return letter if explicit or parenthesized or terminal or choice_surface else None

        # Accept an extracted option surface only if that same surface is
        # actually present in the model response.  Return the canonical choice
        # text and avoid edit-distance guessing on arbitrary strings.
        normalized_raw = _normalized_surface(raw)
        for choice in choices:
            if normalized_raw == _normalized_surface(choice) and _surface_contains(
                response_text, choice
            ):
                return choice
        return None

    if _surface_contains(response_text, raw):
        return raw

    if str(problem.get("answer_type", "")) == "list":
        elements = re.findall(r"[-+]?\d+(?:\.\d+)?|[A-Za-z]+", raw)
        if elements and all(_surface_contains(response_text, item) for item in elements):
            return raw
    return None


def _quick_extract(response: str, problem: dict) -> str | None:
    """Try rule-based extraction before falling back to LLM."""
    if not response:
        return ""
    question_type = problem.get("question_type", "")
    answer_type = problem.get("answer_type", "")
    choices = problem.get("choices") or []

    if question_type == "multi_choice" and response in choices:
        return response
    if answer_type == "integer":
        try:
            return str(int(response))
        except (ValueError, TypeError):
            pass
    if answer_type == "float":
        try:
            return str(float(response))
        except (ValueError, TypeError):
            pass

    # Try regex for "Final answer: ..." or "Answer: ..."
    match = re.search(r"(?:Final answer:|Answer:)\s*(.*)", response, re.IGNORECASE)
    if match:
        ans = match.group(1).strip()
        if ans:
            return ans

    return None  # need LLM extraction


def _build_extract_prompt(problem: dict[str, Any], response: str) -> str:
    """Build a response-only prompt so the judge cannot solve the task."""

    question_type = str(problem.get("question_type", "unknown"))
    answer_type = str(problem.get("answer_type", "unknown"))
    choices = problem.get("choices") or []
    choice_text = ""
    if question_type == "multi_choice" and choices:
        choice_text = "\nValid choices:\n" + "\n".join(
            f"({chr(ord('A') + index)}) {choice}"
            for index, choice in enumerate(choices)
        )
    target = (
        "Extract only an answer explicitly stated in the target model response. "
        "Do not solve or guess the original problem. If the response does not "
        f"state an answer, output {INVALID_EXTRACTION}.\n"
        f"Question type: {question_type}\nAnswer type: {answer_type}"
        f"{choice_text}\n\nTarget model response:\n{response}\n\n"
        "Extracted answer: "
    )
    return f"{_EXTRACT_DEMO_PROMPT.strip()}\n\n{target}"


def _run_llm_extraction(
    results: dict[str, Any],
    model_path: str,
    max_new_tokens: int = 256,
    use_quick_extract: bool = True,
) -> dict[str, Any]:
    """Extract answers from model responses using a local LLM (e.g. Qwen3-32B)."""
    # First pass: check which items already have extraction or can be rule-extracted
    already_done = 0
    need_llm = []
    for pid, problem in results.items():
        if "choices" not in problem:
            continue
        # Revalidate prior extractions as well.  Older local-judge results may
        # have answered the target question instead of extracting the response.
        if "extraction" in problem and problem["extraction"]:
            if problem["extraction"] == INVALID_EXTRACTION:
                already_done += 1
                continue
            grounded = _ground_extraction(
                problem.get("response", ""), problem["extraction"], problem
            )
            if grounded is None:
                problem.setdefault("raw_extraction", problem["extraction"])
                problem["extraction"] = INVALID_EXTRACTION
                problem["extraction_grounded"] = False
            else:
                problem["extraction"] = grounded
                problem["extraction_grounded"] = True
            already_done += 1
            continue
        response = problem.get("response", "")
        if use_quick_extract:
            quick = _quick_extract(response, problem)
            if quick is not None:
                grounded = _ground_extraction(response, quick, problem)
                if grounded is None:
                    problem["raw_extraction"] = quick
                    problem["extraction"] = INVALID_EXTRACTION
                    problem["extraction_grounded"] = False
                else:
                    problem["extraction"] = grounded
                    problem["extraction_grounded"] = True
                continue
        need_llm.append(pid)

    if already_done > 0:
        print(
            f"[mathvista] {already_done} already extracted from previous run",
            flush=True,
        )

    print(
        f"[mathvista] {len(results) - len(need_llm) - already_done} extracted by rules, "
        f"{len(need_llm)} need LLM",
        flush=True,
    )

    # Only load the heavy LLM if there are items that actually need it
    if not need_llm:
        print("[mathvista] all items already extracted, skipping LLM load", flush=True)
        return results

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[mathvista] loading extraction LLM: {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    for pid in tqdm(need_llm, desc="mathvista/extract", file=sys.stdout):
        problem = results[pid]
        response = problem.get("response", "")
        prompt = _build_extract_prompt(problem, response)

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        extraction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        # Handle Qwen3 thinking format: strip <think>...</think> block
        think_match = re.search(r"</think>\s*(.*)", extraction, re.DOTALL)
        if think_match:
            extraction = think_match.group(1).strip()
        # Take only the first line as the extracted answer
        extraction = extraction.split("\n")[0].strip()
        grounded = _ground_extraction(response, extraction, problem)
        if grounded is None:
            problem["raw_extraction"] = extraction
            problem["extraction"] = INVALID_EXTRACTION
            problem["extraction_grounded"] = False
        else:
            problem["extraction"] = grounded
            problem["extraction_grounded"] = True

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[mathvista] extraction done for {len(results)} items", flush=True)
    return results


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
    }
    return aliases.get(normalized, normalized)


def _extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("text", "answer", "response", "output", "generated_text"):
            value = output.get(key)
            if isinstance(value, str):
                return value
        results = output.get("results")
        if isinstance(results, dict):
            for key in ("text", "answer", "response", "output"):
                value = results.get(key)
                if isinstance(value, str):
                    return value
        if isinstance(results, list):
            for item in results:
                text = _extract_text(item)
                if text:
                    return text
        # Handle adapters that return {"understandings": [{"response": "..."}]}
        for list_key in ("understandings",):
            container = output.get(list_key)
            if isinstance(container, list):
                for item in container:
                    text = _extract_text(item)
                    if text:
                        return text
    if isinstance(output, list):
        for item in output:
            text = _extract_text(item)
            if text:
                return text
    return ""


def _load_eval_cfg(config_path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_cfg = load_config(config_path)
    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    mathvista_cfg = raw_cfg.get("mathvista", {}) if isinstance(raw_cfg.get("mathvista"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}
    if not eval_cfg and "benchmark" in raw_cfg:
        eval_cfg = {"benchmark": raw_cfg.get("benchmark")}
    return eval_cfg, mathvista_cfg, inference_cfg


def _find_latest_results(out_dir: Path, ds_name: str) -> Path | None:
    """Find the most recent results JSON for a dataset in out_dir."""
    candidates = sorted(out_dir.glob(f"{ds_name}_*.json"))
    candidates = [c for c in candidates if "_checkpoint" not in c.name and "_score" not in c.name]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def run_mathvista_eval_command(args: Any) -> int:
    config_path = str(args.config)
    eval_cfg, mathvista_cfg, inference_cfg = _load_eval_cfg(config_path)
    benchmark = str(eval_cfg.get("benchmark", "")).strip().lower()
    if benchmark != "mathvista":
        raise ValueError(f"Expected `eval.benchmark: mathvista`, got: {benchmark or '<empty>'}")

    repo_root = Path(__file__).resolve().parents[3]

    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required for MathVista eval.")
    backbone = _normalize_backbone_name(backbone_raw)

    backbone_cfg = inference_cfg.get("backbone_cfg", {})
    if not isinstance(backbone_cfg, dict):
        raise ValueError("`inference.backbone_cfg` must be a dict when provided.")

    request_cfg = inference_cfg.get("request", {})
    request_params: dict[str, Any] = {}
    if isinstance(request_cfg, dict):
        params = request_cfg.get("params", {})
        if isinstance(params, dict):
            request_params = dict(params)

    datasets_value = mathvista_cfg.get("datasets", ["MathVista_testmini"])
    if isinstance(datasets_value, str):
        datasets = [name.strip() for name in datasets_value.split(",") if name.strip()]
    elif isinstance(datasets_value, list):
        datasets = [str(name).strip() for name in datasets_value if str(name).strip()]
    else:
        datasets = ["MathVista_testmini"]
    if not datasets:
        raise ValueError("`mathvista.datasets` must contain at least one dataset name.")

    out_dir = _resolve_path(str(mathvista_cfg.get("out_dir", f"output/mathvista/{backbone}")), repo_root)
    image_dir = _resolve_path(str(mathvista_cfg.get("image_dir", out_dir / "images")), repo_root)
    score_output_path = mathvista_cfg.get("score_output_path")
    cache_dir = mathvista_cfg.get("cache_dir")
    max_samples = int(mathvista_cfg.get("max_samples", 0) or 0)
    use_cot = bool(mathvista_cfg.get("cot", False))
    gt_file = mathvista_cfg.get("gt_file")
    resume = bool(mathvista_cfg.get("resume", False))

    # Mode support (like wise): generate / score / full
    mode = str(mathvista_cfg.get("mode", "full")).strip().lower()
    if mode not in ("full", "generate", "score"):
        print(f"[mathvista] unknown mode '{mode}', defaulting to 'full'", flush=True)
        mode = "full"

    run_gen = mode in ("full", "generate")
    run_score = mode in ("full", "score")

    # LLM extraction config (replaces OpenAI)
    llm_extract_cfg = mathvista_cfg.get("llm_extract", {})
    if not isinstance(llm_extract_cfg, dict):
        llm_extract_cfg = {}
    llm_model_path = str(llm_extract_cfg.get("model_path", "")).strip()
    llm_max_new_tokens = int(llm_extract_cfg.get("max_new_tokens", 2048))

    # Legacy OpenAI config (fallback when llm_extract is not configured)
    run_extract_legacy = bool(mathvista_cfg.get("run_extract", False))
    run_calculation_legacy = bool(mathvista_cfg.get("run_calculation", False))
    openai_api_key = mathvista_cfg.get("openai_api_key")

    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "benchmark": "mathvista",
        "extraction_protocol": EXTRACTION_PROTOCOL,
        "backbone": backbone,
        "out_dir": str(out_dir),
        "datasets": datasets,
        "cot": use_cot,
        "mode": mode,
    }

    # ── Phase 1: Generation ──
    if run_gen:
        from datasets import load_dataset
        from PIL import Image

        from umm.inference import InferencePipeline

        pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

        for ds_name in datasets:
            entry = DS_COLLECTIONS.get(ds_name)
            if not entry:
                raise ValueError(f"Unknown MathVista dataset: {ds_name}")

            dataset_root = str(mathvista_cfg.get("root", entry["root"]))
            split = str(mathvista_cfg.get("split", entry["split"]))
            dataset = load_dataset(
                dataset_root,
                cache_dir=str(_resolve_path(cache_dir, repo_root)) if cache_dir else None,
            )
            data = dataset[split]

            checkpoint_json = out_dir / f"{ds_name}_checkpoint.json"
            results: dict[str, Any] = {}
            results_file: Path | None = None

            if resume:
                if checkpoint_json.exists():
                    results = json.loads(checkpoint_json.read_text("utf-8"))
                    print(f"[mathvista] resume from checkpoint: {len(results)} done", flush=True)
                else:
                    results_file = _find_latest_results(out_dir, ds_name)
                    if results_file:
                        print(f"[mathvista] resume: using completed file {results_file}", flush=True)

            if results_file is None:
                print(
                    f"[mathvista] {ds_name}: {len(data)} total, {len(results)} done, "
                    f"{len(data) - len(results)} remaining",
                    flush=True,
                )
                for idx, data_item in enumerate(tqdm(data, desc=f"mathvista/{ds_name}", file=sys.stdout), start=1):
                    pid = data_item.get("pid")
                    if pid is None:
                        raise ValueError("MathVista sample missing `pid`.")
                    if str(pid) in results:
                        continue

                    image = data_item.get("decoded_image")
                    if image is None:
                        raise ValueError("MathVista sample missing `decoded_image`.")
                    if not isinstance(image, Image.Image):
                        raise ValueError("Expected `decoded_image` to be a PIL image.")

                    image_dir.mkdir(parents=True, exist_ok=True)
                    image_path = image_dir / f"{pid}.png"
                    image.save(image_path, format="PNG")

                    question = data_item.get("query")
                    if question is None:
                        raise ValueError("MathVista sample missing `query`.")
                    if use_cot:
                        prompt = COT_INSTRUCTION.format(question=question)
                    else:
                        prompt = question

                    payload = {
                        "backbone": backbone,
                        "task": "understanding",
                        "prompt": prompt,
                        "images": [str(image_path)],
                        "params": request_params,
                        "metadata": {
                            "pid": pid,
                            "dataset": ds_name,
                            "question_type": data_item.get("question_type"),
                            "choices": {
                                chr(ord("A") + choice_index): str(choice)
                                for choice_index, choice in enumerate(
                                    data_item.get("choices") or []
                                )
                            },
                        },
                    }
                    output = pipeline.run(payload)
                    response = _extract_text(output)

                    item = dict(data_item)
                    item.pop("decoded_image", None)
                    item["response"] = response
                    if isinstance(output, dict) and output.get(
                        "quantization_route_diagnostics"
                    ) is not None:
                        item["quantization_route_diagnostics"] = output[
                            "quantization_route_diagnostics"
                        ]
                    results[str(pid)] = item

                    checkpoint_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

                    if max_samples > 0 and len(results) >= max_samples:
                        break

                time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
                results_file = out_dir / f"{ds_name}_{time_prefix}.json"
                results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

                if checkpoint_json.exists():
                    checkpoint_json.unlink()

            summary[f"{ds_name}_output_path"] = str(results_file)

        if mode == "generate":
            print(f"[mathvista] generation phase done, outputs={out_dir}", flush=True)

        # Free GPU memory from generation pipeline before scoring
        del pipeline
        import gc
        gc.collect()
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        print("[mathvista] released generation pipeline GPU memory", flush=True)

    # ── Phase 2: Scoring (extract + calculate) ──
    if run_score:
        for ds_name in datasets:
            # Find results file from generation phase (or previous run)
            results_file = None
            if f"{ds_name}_output_path" in summary:
                results_file = Path(summary[f"{ds_name}_output_path"])
            else:
                results_file = _find_latest_results(out_dir, ds_name)
            if results_file is None or not results_file.exists():
                raise FileNotFoundError(
                    f"No results file found for {ds_name} in {out_dir}. "
                    f"Run generation phase first (mode: generate)."
                )

            print(f"[mathvista] scoring {ds_name} from {results_file}", flush=True)
            results = json.loads(results_file.read_text("utf-8"))

            # ── Extract answers ──
            if llm_model_path:
                # Use local Qwen model for extraction
                results = _run_llm_extraction(
                    results,
                    model_path=llm_model_path,
                    max_new_tokens=llm_max_new_tokens,
                    use_quick_extract=use_cot,
                )
                # Save results with extraction field
                results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
                print(f"[mathvista] saved extractions to {results_file}", flush=True)
            elif run_extract_legacy:
                # Fallback: use legacy OpenAI-based extract_answer.py
                cmd = [
                    sys.executable,
                    "src/umm/eval/internvl_chat/eval/mathvista/extract_answer.py",
                    "--output_file",
                    results_file.name,
                    "--output_dir",
                    str(out_dir),
                ]
                if use_cot:
                    cmd.append("--quick_extract")
                env = None
                if isinstance(openai_api_key, str) and openai_api_key.strip():
                    env = dict(os.environ)
                    env["OPENAI_API_KEY"] = openai_api_key.strip()
                proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, env=env)
                print(proc.stdout)
                if proc.returncode != 0:
                    if proc.stderr:
                        print(proc.stderr, file=sys.stderr)
                    raise RuntimeError(f"MathVista extract_answer failed with return code {proc.returncode}")
                summary[f"{ds_name}_extract_stdout"] = proc.stdout

            # ── Calculate scores ──
            score_file = results_file.with_name(f"{results_file.stem}_score.json")
            cmd = [
                sys.executable,
                "src/umm/eval/internvl_chat/eval/mathvista/calculate_score.py",
                "--output_file",
                results_file.name,
                "--output_dir",
                str(out_dir),
                "--score_file",
                score_file.name,
            ]
            if isinstance(gt_file, str) and gt_file.strip():
                cmd.extend(["--gt_file", gt_file.strip()])
            proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
            print(proc.stdout)
            if proc.returncode != 0:
                if proc.stderr:
                    print(proc.stderr, file=sys.stderr)
                raise RuntimeError(f"MathVista calculate_score failed with return code {proc.returncode}")
            score_payload = json.loads(score_file.read_text("utf-8"))
            score_payload["extraction_protocol"] = EXTRACTION_PROTOCOL
            score_file.write_text(json.dumps(score_payload, indent=2), encoding="utf-8")
            summary[f"{ds_name}_score_file"] = str(score_file)
            summary[f"{ds_name}_score_stdout"] = proc.stdout

    if isinstance(score_output_path, str) and score_output_path:
        score_path = _resolve_path(score_output_path, repo_root)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[umm eval] wrote MathVista summary to {score_path}")

    print(f"[umm eval] completed MathVista (mode={mode}) for backbone={backbone}, outputs={out_dir}")
    return 0
