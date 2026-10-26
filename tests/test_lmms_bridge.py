"""CPU contract tests for the UMM adapter using the installed lmms base class.

Set PYTHONPATH to TorchUMM/src and the selected lmms-eval checkout; no model,
dataset downloads, CUDA initialization, or scorer monkeypatches are needed.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

try:
    from lmms_eval.api.model import lmms
    from umm.eval.lmms_bridge import UMMGenerationAdapter
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"Optional lmms runtime unavailable: {exc.name}") from exc


class RecordingPipeline:
    backbone_name = "janus_pro"

    def __init__(self, answers=None, error=None):
        self.calls = []
        self.answers = iter(answers or [])
        self.error = error
        self.loaded_model = object()

    def load(self, *args, **kwargs):
        raise AssertionError("An injected pipeline must never reload its model")

    def run(self, payload):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return next(self.answers)


def request(context, kwargs=None, doc_id=0, task="fixture", split="test"):
    return SimpleNamespace(
        args=(context, kwargs or {}, lambda doc: doc["images"], doc_id, task, split)
    )


class UMMGenerationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="umm-lmms-contract-")
        self.root = Path(self.temp.name)
        self.raw = self.root / "generations.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def adapter(self, pipeline, documents=None, **kwargs):
        adapter = UMMGenerationAdapter(
            pipeline=pipeline,
            raw_output_path=self.raw,
            image_dir=self.root / "images",
            generation_defaults=kwargs.pop("generation_defaults", {"max_new_tokens": 128}),
            generation_overrides=kwargs.pop("generation_overrides", {}),
            model_identity=kwargs.pop(
                "model_identity", {"arm": "fake-quantized", "checkpoint": "fixture-only"}
            ),
            **kwargs,
        )
        adapter.task_dict = {"fixture": {"test": documents or [{"images": []}]}}
        return adapter

    def records(self):
        return [json.loads(line) for line in self.raw.read_text().splitlines()]

    def test_real_lmms_base_and_injected_pipeline_identity(self):
        pipeline = RecordingPipeline([{"text": "answer"}])
        loaded_model = pipeline.loaded_model
        adapter = self.adapter(pipeline)
        self.assertIsInstance(adapter, lmms)
        self.assertTrue(adapter.is_simple)
        self.assertEqual((adapter.rank, adapter.world_size), (0, 1))
        self.assertEqual(adapter.generate_until([request("question")]), ["answer"])
        adapter.clean()
        self.assertIs(pipeline.loaded_model, loaded_model)
        self.assertIs(adapter.pipeline, pipeline)
        self.assertEqual(len(pipeline.calls), 1)
        self.assertEqual(pipeline.calls[0]["task"], "understanding")
        self.assertEqual(pipeline.calls[0]["backbone"], "janus_pro")

    def test_pixels_and_request_order_are_preserved(self):
        first = Image.new("RGB", (3, 2))
        first.putdata([(13 * i, 255 - 20 * i, 11 * i) for i in range(6)])
        second = Image.new("RGB", (2, 4), (3, 7, 19))
        pipeline = RecordingPipeline([{"text": "second"}, {"text": "first"}])
        adapter = self.adapter(pipeline, [{"images": [first]}, {"images": [second]}])
        result = adapter.generate_until([request("q2", doc_id=1), request("q1", doc_id=0)])
        self.assertEqual(result, ["second", "first"])
        for call, expected in zip(pipeline.calls, [second, first]):
            self.assertEqual(len(call["images"]), 1)
            path = Path(call["images"][0])
            self.assertEqual(path.suffix.lower(), ".png")
            self.assertTrue(path.is_relative_to(self.root / "images"))
            with Image.open(path) as restored:
                self.assertEqual(restored.mode, expected.mode)
                self.assertEqual(restored.size, expected.size)
                self.assertEqual(restored.tobytes(), expected.tobytes())
        self.assertNotEqual(pipeline.calls[0]["images"], pipeline.calls[1]["images"])
        self.assertEqual([call["prompt"] for call in pipeline.calls], ["q2", "q1"])

    def test_task_generation_settings_override_defaults_and_do_not_mutate(self):
        pipeline = RecordingPipeline([{"text": "a"}])
        defaults = {"max_new_tokens": 128, "do_sample": False}
        task_kwargs = {"max_new_tokens": 19, "do_sample": False, "temperature": 0}
        adapter = self.adapter(pipeline, generation_defaults=defaults)
        adapter.generate_until([request("q", task_kwargs)])
        self.assertEqual(pipeline.calls[0]["params"]["max_new_tokens"], 19)
        self.assertIs(pipeline.calls[0]["params"]["do_sample"], False)
        self.assertEqual(defaults, {"max_new_tokens": 128, "do_sample": False})
        self.assertEqual(task_kwargs, {"max_new_tokens": 19, "do_sample": False, "temperature": 0})

    def test_explicit_generation_override_takes_precedence(self):
        pipeline = RecordingPipeline([{"text": "a"}])
        adapter = self.adapter(
            pipeline,
            generation_defaults={"max_new_tokens": 128},
            generation_overrides={"max_new_tokens": 23, "do_sample": False},
        )
        adapter.generate_until([request("q", {"max_new_tokens": 7, "do_sample": True})])
        self.assertEqual(pipeline.calls[0]["params"]["max_new_tokens"], 23)
        self.assertIs(pipeline.calls[0]["params"]["do_sample"], False)

    def test_until_uses_first_text_position_and_retains_raw_answer(self):
        raw_answer = " kept <early> ignored <late> "
        pipeline = RecordingPipeline([{"text": raw_answer}])
        adapter = self.adapter(pipeline)
        result = adapter.generate_until([request("q", {"until": ["<late>", "<early>"]})])
        self.assertEqual(result, [" kept "])
        self.assertNotIn("until", pipeline.calls[0]["params"])
        record = self.records()[0]
        self.assertEqual(record["raw_response"], raw_answer)
        self.assertEqual(record["response"], " kept ")

    def test_empty_answer_is_preserved_without_option_fallback(self):
        adapter = self.adapter(RecordingPipeline([{"text": ""}]))
        self.assertEqual(adapter.generate_until([request("q")]), [""])
        self.assertEqual(self.records()[0]["raw_response"], "")
        self.assertEqual(self.records()[0]["response"], "")

    def test_path_provenance_is_normalized_and_saved_before_generation(self):
        checkpoint = self.root / "quantized-model.pt"
        identity = {"checkpoint": checkpoint, "nested": {"manifest": self.root / "manifest.json"}}
        pipeline = RecordingPipeline([{"text": "answer"}])
        adapter = self.adapter(pipeline, model_identity=identity)
        self.assertEqual(adapter.model_identity["checkpoint"], str(checkpoint))
        self.assertEqual(pipeline.calls, [])
        self.assertEqual(adapter.generate_until([request("q")]), ["answer"])
        saved = self.records()[0]["model_identity"]
        self.assertEqual(saved["checkpoint"], str(checkpoint))
        self.assertEqual(saved["nested"]["manifest"], str(self.root / "manifest.json"))
        self.assertIsInstance(identity["checkpoint"], Path)

    def test_choice_dependent_dynamic_pipelines_are_rejected_before_inference(self):
        backbones = [
            SimpleNamespace(quantization_prefill_selector=True),
            SimpleNamespace(operator_quotient_routes={"understanding": "confidence_cayley_prefill_decode_baseline"}),
            SimpleNamespace(operator_quotient_routes={"understanding": "unanimous_cayley_choice_fallback_baseline"}),
        ]
        for backbone in backbones:
            with self.subTest(backbone=backbone):
                pipeline = RecordingPipeline([{"text": "not reached"}])
                pipeline.backbone = backbone
                with self.assertRaisesRegex(NotImplementedError, "static|dynamic|metadata"):
                    self.adapter(pipeline)
                self.assertEqual(pipeline.calls, [])

    def test_pipeline_route_diagnostics_are_retained_in_raw_record(self):
        diagnostics = {
            "operator_route": "baseline",
            "operator_route_diagnostics": {"selected": "baseline", "reason": "static"},
            "route_diagnostics": {"wrapper": "static-quant-fixture"},
            "quantization_route_diagnostics": {"selected": "understanding_static", "layers": 168},
        }
        adapter = self.adapter(RecordingPipeline([{"text": "answer", **diagnostics}]))
        self.assertEqual(adapter.generate_until([request("q")]), ["answer"])
        self.assertEqual(self.records()[0]["pipeline_diagnostics"], diagnostics)

    def test_pipeline_failure_is_not_replaced_by_empty_answer(self):
        adapter = self.adapter(RecordingPipeline(error=RuntimeError("quantization key miss")))
        with self.assertRaisesRegex(RuntimeError, "quantization key miss"):
            adapter.generate_until([request("q")])

    def test_missing_output_text_is_a_protocol_error(self):
        adapter = self.adapter(RecordingPipeline([{"loss": 0.5}]))
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            adapter.generate_until([request("q")])

    def test_unknown_generation_key_is_rejected_before_inference(self):
        pipeline = RecordingPipeline([{"text": "not reached"}])
        adapter = self.adapter(pipeline)
        with self.assertRaises((TypeError, ValueError, NotImplementedError)):
            adapter.generate_until([request("q", {"silently_ignored_parameter": 1})])
        self.assertEqual(pipeline.calls, [])

    def test_sampling_is_rejected_when_backbones_cannot_honor_parameters(self):
        pipeline = RecordingPipeline([{"text": "not reached"}])
        adapter = self.adapter(pipeline)
        with self.assertRaises((TypeError, ValueError, NotImplementedError)):
            adapter.generate_until([request("q", {"do_sample": True, "temperature": 0.7})])
        self.assertEqual(pipeline.calls, [])

    def test_non_neutral_sampling_parameter_is_rejected(self):
        pipeline = RecordingPipeline([{"text": "not reached"}])
        adapter = self.adapter(pipeline)
        with self.assertRaises((TypeError, ValueError, NotImplementedError)):
            adapter.generate_until([request("q", {"do_sample": False, "temperature": 0.7})])
        self.assertEqual(pipeline.calls, [])

    def test_unsupported_task_methods_are_explicit(self):
        adapter = self.adapter(RecordingPipeline())
        with self.assertRaises(NotImplementedError):
            adapter.loglikelihood([])
        with self.assertRaises(NotImplementedError):
            adapter.generate_until_multi_round([])

    def test_multiple_images_are_rejected_before_single_image_backbone_call(self):
        # Janus/Emu3.5 currently consume images[0]. A bridge must not silently
        # discard extra visuals, even when these four benchmark tasks use one.
        pipeline = RecordingPipeline([{"text": "not reached"}])
        adapter = self.adapter(pipeline, [{"images": [Image.new("RGB", (2, 2))] * 2}])
        with self.assertRaises((ValueError, NotImplementedError)):
            adapter.generate_until([request("q")])
        self.assertEqual(pipeline.calls, [])


if __name__ == "__main__":
    unittest.main()
