#!/usr/bin/env bash

cd "$(dirname "$0")/.." || exit 1

BASE_MODEL_PATH="weights/DeViL-7B-stage2"
LORA_CHECKPOINT_PATH="weights/DeViL-7B-stage3/checkpoint-23688"
OUTPUT_PATH="weights/DeViL-7B"

python demo/merge_lora.py \
  --base_model_path "${BASE_MODEL_PATH}" \
  --lora_checkpoint_path "${LORA_CHECKPOINT_PATH}" \
  --output_path "${OUTPUT_PATH}"
