# lmms-eval tasks through UMM

This optional bridge reuses lmms-eval's dataset loading, task prompts and scoring,
and sends each `generate_until` request to UMM's understanding pipeline. It is a
separate protocol from the existing UMM benchmark configurations. The initial
templates cover Janus-Pro and Emu3.5 with these upstream task names:

| Task | Scope |
| --- | --- |
| `ocrbench` | OCRBench v1 |
| `chartqa` | ChartQA, including its upstream human/augmented scoring |
| `mmerealworld` | MME-RealWorld English full |
| `textvqa_val` | TextVQA validation, upstream multi-annotation soft score |

Replace `mmerealworld` with `mmerealworld_lite` for English Lite or
`mmerealworld_cn` for Chinese. The dataset's published split is named `train`;
the bridge preserves that upstream convention. This is distinct from MME.

## Source and environment

Use an unchanged lmms-eval checkout at
`8f40097532a7bf4db6123713f1a905e02c8f5971`. Set `LMMS_EVAL_ROOT` to its absolute
path. The runner verifies the actual imported package/evaluator paths and Git
revision, and rejects local changes under `lmms_eval` when a revision is pinned.
The active Python environment needs that checkout's task dependencies as well as
the existing model environment. The bridge imports these dependencies lazily;
unrelated UMM commands do not require lmms-eval. It does not install dependencies
or download model weights itself. Upstream tasks can download their datasets.

After configuring `UMM_MODEL_CACHE`, `UMM_CODEBASE` and the existing backbone
environment, copy a template and choose a **new** `lmms.out_dir`:

```bash
PYTHONPATH=src python -m umm.cli.main eval \
  --config configs/eval/lmms/lmms_janus_pro.yaml
```

The Emu3.5 template is `configs/eval/lmms/lmms_emu3_5.yaml`. Model loading follows
the corresponding existing UMM backbone. Its tensor parallelism and image size
are explicit in the template. These are configuration templates, not a record
of successful GPU evaluation or a new frozen research protocol.

For a cheap integration check set `lmms.limit: 2` in a copied configuration.
Set `lmms.bootstrap_iters: 0` to skip confidence-interval bootstrapping for that
check; the normal default remains upstream's 100000 iterations.
Use `null` for full task coverage. **OCRBench's upstream aggregate divides by
1000 even with a limit:** a limited-run score is not subset accuracy and must
not be reported as a full benchmark result.

Downloaded Parquet snapshots can be selected without modifying upstream task
source. For example:

```yaml
lmms:
  task_overrides:
    chartqa:
      dataset_path: parquet
      dataset_kwargs:
        data_files:
          test: /absolute/path/to/pinned/chartqa/test-*.parquet
```

Only `dataset_path` and `dataset_kwargs` are allowed in these per-task overrides.
The upstream prompt, split, answer processing and scorer remain selected by the
original task. Preserve each task's expected split name and schema; record the
local snapshot revision and checksums in its dataset manifest. Overrides and
resolved task configurations are included in the run outputs. Upstream scorer
submission/report files are directed inside the same fresh run directory.

## Reuse an already quantized model

Scientific callers can inject an existing pipeline without loading or quantizing
another model:

```python
from umm.cli.lmms_eval import run_lmms_evaluation

result = run_lmms_evaluation(
    "configs/eval/lmms/my_fresh_attempt.yaml",
    pipeline=already_loaded_quantized_pipeline,
    model_identity={
        "state_id": "E-ID/Axx/arm-name",
        "checkpoint_sha256": "actual-checkpoint-or-manifest-digest",
        "quantization_manifest": "/absolute/path/to/manifest.json",
    },
)
```

An injected pipeline must provide the same UMM `backbone_name` and `run(payload)`
understanding interface. This initial adapter supports single-image, greedy,
static pipelines. Choice-dependent dynamic selectors are rejected because they
require extra metadata. Neutral task sampling defaults are recorded; unsupported
non-neutral sampling controls raise an error. The identity is caller-declared provenance; it does not
prove that quantization is active. Keep the caller's independent intervention
and coverage checks. `inference.backbone_cfg` is only used to build a pipeline
when none is injected. CLI template identities describe FP loading from a model
path and must be changed when evaluating another state.

## Generation and scoring provenance

Upstream task prompts and generation kwargs take precedence over
`lmms.generation_defaults`. The templates supply a 128-token fallback and greedy
decoding only when a task leaves those fields unspecified. In particular, a
task's 16-token budget is preserved. `lmms.generation_overrides` deliberately
overrides task kwargs; nonempty values are a protocol change recorded in the
manifest. The adapter records effective per-request parameters and retains
responses before and after its stop-string handling. It does not substitute
the prompts or budgets from UMM's old evaluation panel.

Response caches and request caches are disabled. Every run needs a fresh output
directory, and each model arm should use its own process to isolate upstream
task state. Outputs include `manifest.json` (source identity, tasks, model
identity, generation settings and run status), `config.json`, raw
`generations.jsonl`, `samples/<task>.jsonl`, and upstream `results.json` including
resolved task configurations. A failed run retains its manifest and any raw
responses already written.

The pinned MME-RealWorld scorer has a known parser behavior: an empty response
is parsed as `A`, and a response such as `A or B` can be parsed as `A`. This was
checked by executing the upstream parser function in a CPU probe. The bridge
preserves the scorer and raw answers; it does not silently fix or relabel its
scores. Any corrected parser should be a separately named, explicit protocol.
