"""
Modal Volume definitions for umm_codebase.

Four persistent volumes:
  1. umm-model-cache    — HuggingFace model weights (download once, reuse forever)
  2. umm-datasets-cache — HuggingFace datasets (download once, update as needed)
  3. umm-checkpoints    — Training outputs (checkpoints, logs, results)
  4. umm-outputs        — Inference generated results & evaluation scores

Usage:
    from volumes import model_volume, dataset_volume, checkpoint_volume, output_volume
"""

import modal

from config import (
    CHECKPOINT_VOLUME_NAME,
    CODEBASE_VOLUME_NAME,
    DATASET_VOLUME_NAME,
    MODEL_VOLUME_NAME,
    OUTPUT_APR4_VOLUME_NAME,
    OUTPUT_VOLUME_NAME,
    POST_TRAIN_MODEL_VOLUME_NAME,
)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
post_train_model_volume = modal.Volume.from_name(POST_TRAIN_MODEL_VOLUME_NAME, create_if_missing=True)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
output_apr4_volume = modal.Volume.from_name(OUTPUT_APR4_VOLUME_NAME, create_if_missing=True)
codebase_volume = modal.Volume.from_name(CODEBASE_VOLUME_NAME, create_if_missing=True)
