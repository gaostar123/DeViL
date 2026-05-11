import os
import sys
import argparse

import torch
from transformers import AutoTokenizer
from peft import PeftModel

sys.path.append("./")
from devil.constants import DEFAULT_G_DINO_CONFIG_PATH
from devil.model import DeViLQwen2ForCausalLM, DeViLQwen2Config


def merge_lora_to_base_model(base_model_path, lora_checkpoint_path, output_path):
    print("========== Merge LoRA to Base Model ==========")
    print(f"[INFO] Base model path      : {base_model_path}")
    print(f"[INFO] LoRA checkpoint path : {lora_checkpoint_path}")
    print(f"[INFO] Output path          : {output_path}")

    print("\n[STEP 1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        lora_checkpoint_path,
        trust_remote_code=True
    )

    print("[STEP 2] Loading config from base model...")
    config = DeViLQwen2Config.from_pretrained(base_model_path)
    if not hasattr(config, "use_gdino"):
        setattr(config, "use_gdino", True)
    config.g_dino_config_path = getattr(config, "g_dino_config_path", DEFAULT_G_DINO_CONFIG_PATH) or DEFAULT_G_DINO_CONFIG_PATH

    print("[STEP 3] Loading base model...")
    base_model = DeViLQwen2ForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.float16,
    )

    non_lora_path = os.path.join(lora_checkpoint_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        print("[STEP 4] Loading non-LoRA trainable weights...")
        non_lora_state = torch.load(non_lora_path, map_location="cpu")

        non_lora_state = {
            (k[11:] if k.startswith("base_model.") else k): v
            for k, v in non_lora_state.items()
        }
        if any(k.startswith("model.model.") for k in non_lora_state):
            non_lora_state = {
                (k[6:] if k.startswith("model.") else k): v
                for k, v in non_lora_state.items()
            }

        missing, unexpected = base_model.load_state_dict(non_lora_state, strict=False)
        print(f"[INFO] non-LoRA weights loaded. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    else:
        print("[STEP 4] non_lora_trainables.bin not found, skip non-LoRA weights.")

    print("[STEP 5] Loading LoRA adapter and merging...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_checkpoint_path,
    )
    model = model.merge_and_unload()
    print("[INFO] LoRA weights merged into base model.")

    print("[STEP 6] Saving merged model...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=False)
    tokenizer.save_pretrained(output_path)

    print("\n========== Done ==========")
    print(f"[INFO] Merged model saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser("Merge LoRA checkpoint into base model")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to base model (e.g., stage2 llm_lo_merge).",
    )
    parser.add_argument(
        "--lora_checkpoint_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory (e.g., checkpoint-XXXXX).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to save the merged full model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_lora_to_base_model(
        base_model_path=args.base_model_path,
        lora_checkpoint_path=args.lora_checkpoint_path,
        output_path=args.output_path,
    )
