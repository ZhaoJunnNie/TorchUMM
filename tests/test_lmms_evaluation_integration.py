"""Exercise real upstream tasks/evaluator with tiny offline data and a fake pipeline.

Run with LMMS_EVAL_ROOT pointing at the pinned checkout. No weights, remote
datasets, or GPU are used; this verifies integration rather than model quality.
"""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("LMMS_EVAL_ROOT"), "set LMMS_EVAL_ROOT for the real upstream integration test")
class RealLmmsIntegrationTest(unittest.TestCase):
    def test_four_upstream_tasks_preserve_pipeline_and_write_isolated_results(self):
        import datasets
        from PIL import Image

        from umm.cli.lmms_eval import LMMS_EVAL_REVISION, run_lmms_evaluation

        image = Image.new("RGB", (7, 5), (29, 61, 107))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_feature = datasets.Image()

        def image_dataset(rows):
            dataset = datasets.Dataset.from_list([
                {**row, "image": {"bytes": image_bytes, "path": None}} for row in rows
            ])
            return dataset.cast_column("image", image_feature)

        fixtures = {
            "echo840/OCRBench": datasets.DatasetDict({"test": image_dataset([
                {"question": "fixture-ocr-1", "answer": ["sample"], "dataset": "fixture", "question_type": "Regular Text Recognition"},
                {"question": "fixture-ocr-2", "answer": ["other"], "dataset": "fixture", "question_type": "Regular Text Recognition"},
            ])}),
            "lmms-lab-encoder/ChartQA": datasets.DatasetDict({"test": image_dataset([
                {"question": "fixture-chart-1", "answer": "10", "type": "human_test"},
                {"question": "fixture-chart-2", "answer": "10", "type": "augmented_test"},
            ])}),
            "lmms-lab-encoder/textvqa": datasets.DatasetDict({"validation": image_dataset([
                {"question": "fixture-text-1", "question_id": 1, "answer": "sign", "answers": ["sign"] * 10},
                {"question": "fixture-text-2", "question_id": 2, "answer": "sign", "answers": ["sign"] * 10},
            ])}),
            "yifanzhang114/MME-RealWorld-Lmms-eval": datasets.DatasetDict({"train": datasets.Dataset.from_list([
                {"question": "fixture-mme-1", "index": 1, "bytes": base64.b64encode(image_bytes).decode(), "multi-choice options": ["(A) First", "(B) Second", "(C) Third", "(D) Fourth", "(E) Fifth"], "answer": "B", "category": "Perception/OCR with Complex Context", "l2-category": "text"},
                {"question": "fixture-mme-2", "index": 2, "bytes": base64.b64encode(image_bytes).decode(), "multi-choice options": ["(A) First", "(B) Second", "(C) Third", "(D) Fourth", "(E) Fifth"], "answer": "A", "category": "Perception/OCR with Complex Context", "l2-category": "text"},
            ])}),
        }
        loads = []
        original_load_dataset = datasets.load_dataset

        def offline_load_dataset(path, *args, **kwargs):
            loads.append(path)
            if path == "parquet":
                return original_load_dataset(path, *args, **kwargs)
            if path not in fixtures:
                raise AssertionError(f"Unexpected dataset/network request: {path}")
            return fixtures[path]

        class InjectedPipeline:
            backbone_name = "fake_quant_fixture"

            def __init__(self):
                self.calls = []
                self.backbone = object()

            def run(self, payload):
                self.calls.append(payload)
                self.assert_pixels(payload["images"])
                prompt = payload["prompt"].lower()
                if "fixture-ocr" in prompt:
                    return {"text": "sample"}
                if "fixture-chart" in prompt:
                    return {"text": "10"}
                if "fixture-text" in prompt:
                    return {"text": "sign"}
                if "fixture-mme" in prompt:
                    return {"text": "B"}
                raise AssertionError(f"Unexpected task prompt: {prompt}")

            @staticmethod
            def assert_pixels(paths):
                if len(paths) != 1:
                    raise AssertionError(paths)
                with Image.open(paths[0]) as actual:
                    if actual.size != image.size or actual.tobytes() != image.tobytes():
                        raise AssertionError("Image bridge changed dimensions or pixels")

        pipeline = InjectedPipeline()
        original_backbone = pipeline.backbone
        with tempfile.TemporaryDirectory(prefix="umm-real-lmms-") as temporary:
            out_dir = Path(temporary) / "run"
            chart_file = Path(temporary) / "chartqa-test.parquet"
            fixtures["lmms-lab-encoder/ChartQA"]["test"].to_parquet(chart_file)
            config = {
                "eval": {"benchmark": "lmms"},
                "inference": {"backbone": pipeline.backbone_name},
                "lmms": {
                    "source_path": str(Path(os.environ["LMMS_EVAL_ROOT"]).resolve()),
                    "revision": LMMS_EVAL_REVISION,
                    "tasks": ["ocrbench", "chartqa", "mmerealworld", "textvqa_val"],
                    "out_dir": str(out_dir),
                    "limit": 2,
                    "bootstrap_iters": 0,
                    "task_overrides": {
                        "chartqa": {
                            "dataset_path": "parquet",
                            "dataset_kwargs": {
                                "data_files": {"test": str(chart_file)},
                                "cache_dir": str(Path(temporary) / "datasets-cache"),
                                "token": False,
                            },
                        },
                    },
                    "generation_defaults": {"max_new_tokens": 128, "do_sample": False},
                },
            }
            with patch.object(datasets, "load_dataset", side_effect=offline_load_dataset):
                result = run_lmms_evaluation(config, pipeline=pipeline, model_identity={"state_id": "synthetic-quant-injection-fixture"})
            self.assertIs(pipeline.backbone, original_backbone)
            self.assertEqual(len(pipeline.calls), 8)
            self.assertEqual(set(loads), (set(fixtures) - {"lmms-lab-encoder/ChartQA"}) | {"parquet"})
            self.assertTrue(result["manifest"]["source"]["pinned_and_clean"])
            self.assertTrue(result["manifest"]["pipeline_injected"])
            self.assertEqual(result["manifest"]["status"], "completed")
            metrics = result["results"]["results"]
            self.assertEqual(metrics["ocrbench"]["ocrbench_accuracy,none"], 0.001)
            self.assertEqual(metrics["chartqa"]["relaxed_overall,none"], 1.0)
            self.assertEqual(metrics["mmerealworld"]["mme_realworld_score,none"], 0.5)
            self.assertEqual(metrics["textvqa_val"]["exact_match,none"], 1.0)
            raw = [json.loads(line) for line in (out_dir / "generations.jsonl").read_text().splitlines()]
            self.assertEqual(len(raw), 8)
            self.assertTrue((out_dir / "results.json").is_file())
            self.assertTrue((out_dir / "manifest.json").is_file())
            submissions = list(out_dir.rglob("ocrbench_results.txt")) + list(out_dir.rglob("textvqa_submission_*.json"))
            self.assertEqual(len(submissions), 2)
            for submission in submissions:
                self.assertTrue(submission.is_relative_to(out_dir))
            budgets = [call["params"]["max_new_tokens"] for call in pipeline.calls]
            self.assertEqual(sorted(budgets), [16, 16, 16, 16, 128, 128, 128, 128])
            with self.assertRaises(FileExistsError):
                run_lmms_evaluation(config, pipeline=pipeline, model_identity={"state_id": "same"})


if __name__ == "__main__":
    unittest.main()
