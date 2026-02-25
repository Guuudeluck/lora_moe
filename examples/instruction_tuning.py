"""
Example: Instruction Tuning with Alpaca Format

This example demonstrates how to finetune a language model on instruction-following
tasks using the Alpaca data format.

Data format:
{
    "instruction": "What is the capital of France?",
    "input": "",  # Optional context
    "output": "The capital of France is Paris."
}
"""

import os
from llama_finetune.configs.hyperparameters import HYPERPARAMETERS_INSTRUCTION, create_custom_config
from llama_finetune.configs.config import Config, StorageConfig, RayConfig
from llama_finetune.workflows.finetune_pipeline import run_finetuning_workflow


def main():
    # ==================== Configure Storage ====================
    # Option 1: Use environment variables (recommended)
    # Set these before running:
    #   export STORAGE_BACKEND=s3
    #   export STORAGE_BASE_PATH=s3://your-bucket/models
    #   export AWS_ACCESS_KEY_ID=...
    #   export AWS_SECRET_ACCESS_KEY=...

    # Option 2: Configure programmatically
    storage_config = StorageConfig(
        backend="s3",  # or "local" for local filesystem
        base_path="s3://your-bucket/llm-finetune",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_region="us-east-1",
    )

    ray_config = RayConfig(
        address=None,  # None for local mode, or "ray://cluster:10001" for cluster
        num_gpus=4,    # Number of GPUs to use
    )

    config = Config(
        storage=storage_config,
        ray=ray_config,
        project_name="instruction-tuning",
        experiment_name="alpaca-llama2-7b",
    )

    # ==================== Configure Training ====================
    training_config = create_custom_config(
        # Model
        MODEL_ID="llama2-7b-instruct-v1",
        BASE_MODEL="meta-llama/Llama-2-7b-hf",
        MODEL_DESCRIPTION="Llama-2 7B finetuned on instruction-following tasks",

        # Data
        INPUT_DATA_PATH="s3://your-bucket/data/alpaca_data.json",
        DATASET_FORMAT="alpaca",
        MAX_SEQ_LENGTH=512,
        VALIDATION_SPLIT=0.1,
        MAX_SAMPLES=None,  # Use all data, or set a number to limit

        # Finetuning method
        FINETUNING_TYPE="lora",
        LORA_RANK=8,
        LORA_ALPHA=16,
        LORA_DROPOUT=0.05,
        LORA_TARGET_MODULES=["q_proj", "v_proj", "k_proj", "o_proj"],

        # Training hyperparameters
        BATCH_SIZE=4,
        GRADIENT_ACCUMULATION_STEPS=4,
        LEARNING_RATE=2e-4,
        NUM_EPOCHS=3,
        WARMUP_RATIO=0.1,
        LR_SCHEDULER_TYPE="cosine",

        # Hardware
        NUM_GPUS=4,
        USE_DEEPSPEED=False,
        USE_BF16=True,
        GRADIENT_CHECKPOINTING=False,

        # Logging and checkpointing
        LOGGING_STEPS=10,
        SAVE_STEPS=500,
        EVAL_STEPS=500,
        SAVE_TOTAL_LIMIT=3,
    )

    # ==================== Run Training ====================
    print("=" * 80)
    print("Instruction Tuning Example - Alpaca Format")
    print("=" * 80)
    print(f"Model: {training_config['BASE_MODEL']}")
    print(f"Data: {training_config['INPUT_DATA_PATH']}")
    print(f"Method: {training_config['FINETUNING_TYPE'].upper()}")
    print("=" * 80)

    results = run_finetuning_workflow(training_config, config)

    # ==================== Results ====================
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Model Path: {results['model_path']}")
    print(f"Adapter Path: {results.get('adapter_path', 'N/A')}")
    print(f"Train Samples: {results['num_train_samples']}")
    print(f"Val Samples: {results['num_val_samples']}")
    print(f"Train Loss: {results['metrics'].get('train_loss', 'N/A')}")
    print(f"Eval Loss: {results['metrics'].get('eval_loss', 'N/A')}")
    print("=" * 80)

    # ==================== Test the Model ====================
    print("\nTo test the model:")
    print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained("{training_config['BASE_MODEL']}")
model = PeftModel.from_pretrained(base_model, "{results.get('adapter_path', results['model_path'])}")
tokenizer = AutoTokenizer.from_pretrained("{results['model_path']}")

# Generate
prompt = "### Instruction:\\nWhat is machine learning?\\n\\n### Response:\\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
    """)


if __name__ == "__main__":
    main()
