"""
Example: Memory-Efficient QLoRA Finetuning

This example demonstrates how to finetune large models on limited GPU memory
using QLoRA (4-bit quantization + LoRA).

QLoRA can finetune a 7B model on a single 24GB GPU or a 13B model on a single 48GB GPU.
"""

import os
from llama_finetune.configs.hyperparameters import HYPERPARAMETERS_QLORA, create_custom_config
from llama_finetune.workflows.finetune_pipeline import run_finetuning_workflow


def main():
    # ==================== Configure Training ====================
    training_config = create_custom_config(
        # Model
        MODEL_ID="llama2-13b-qlora-v1",
        BASE_MODEL="meta-llama/Llama-2-13b-hf",  # 13B model on limited GPU

        # Data
        INPUT_DATA_PATH="s3://your-bucket/data/training_data.json",
        DATASET_FORMAT="alpaca",
        MAX_SEQ_LENGTH=512,

        # QLoRA Configuration
        FINETUNING_TYPE="qlora",  # 4-bit quantization
        LORA_RANK=64,  # Higher rank to compensate for quantization
        LORA_ALPHA=16,
        LORA_DROPOUT=0.1,

        # Training - Optimized for memory
        BATCH_SIZE=1,  # Small batch size
        GRADIENT_ACCUMULATION_STEPS=16,  # Accumulate to effective batch size of 16
        LEARNING_RATE=2e-4,
        NUM_EPOCHS=3,

        # Hardware - Single GPU!
        NUM_GPUS=1,  # QLoRA works on single GPU
        USE_DEEPSPEED=False,  # Not needed for QLoRA
        USE_BF16=True,
        GRADIENT_CHECKPOINTING=True,  # Save more memory

        # Checkpointing
        SAVE_STEPS=1000,
        EVAL_STEPS=1000,
    )

    # ==================== Run Training ====================
    print("=" * 80)
    print("QLoRA Finetuning Example")
    print("=" * 80)
    print(f"Model: {training_config['BASE_MODEL']}")
    print(f"Method: QLoRA (4-bit quantization)")
    print(f"GPUs: {training_config['NUM_GPUS']}")
    print(f"Memory Optimizations:")
    print(f"  - 4-bit quantization: Enabled")
    print(f"  - Gradient checkpointing: {training_config['GRADIENT_CHECKPOINTING']}")
    print(f"  - Batch size: {training_config['BATCH_SIZE']}")
    print(f"  - Gradient accumulation: {training_config['GRADIENT_ACCUMULATION_STEPS']}")
    print("=" * 80)

    results = run_finetuning_workflow(training_config)

    # ==================== Results ====================
    print("\n" + "=" * 80)
    print("QLoRA Training Complete!")
    print("=" * 80)
    print(f"Model Path: {results['model_path']}")
    print(f"Adapter Path: {results.get('adapter_path', 'N/A')}")
    print("\nNote: The adapter is small (~100MB) and can be loaded with:")
    print(f"""
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "{training_config['BASE_MODEL']}",
    load_in_4bit=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "{results.get('adapter_path')}")
    """)


if __name__ == "__main__":
    main()
