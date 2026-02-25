"""
Example: Local Training (No Cloud Storage)

This example demonstrates how to run training entirely on local filesystem,
without requiring S3 or other cloud storage.

Perfect for:
- Local development and testing
- On-premise deployments
- Air-gapped environments
"""

import os
from pathlib import Path
from llama_finetune.configs.hyperparameters import HYPERPARAMETERS_LORA, create_custom_config
from llama_finetune.configs.config import Config, StorageConfig, RayConfig
from llama_finetune.workflows.finetune_pipeline import run_finetuning_workflow


def main():
    # ==================== Setup Local Paths ====================
    # Define local directories
    project_root = Path.home() / "llm-experiments"
    data_dir = project_root / "data"
    output_dir = project_root / "outputs"

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using local directories:")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")

    # ==================== Configure Local Storage ====================
    config = Config(
        storage=StorageConfig(
            backend="local",
            base_path=str(output_dir),
        ),
        ray=RayConfig(
            address=None,  # Local mode
            num_gpus=1,    # Use 1 GPU locally
        ),
        project_name="local-finetune",
        experiment_name="test-run",
    )

    # ==================== Configure Training ====================
    training_config = create_custom_config(
        # Model - Use smaller model for local testing
        MODEL_ID="llama2-7b-local-test",
        BASE_MODEL="meta-llama/Llama-2-7b-hf",

        # Data - Local file path
        INPUT_DATA_PATH=str(data_dir / "training_data.json"),
        DATASET_FORMAT="alpaca",
        MAX_SEQ_LENGTH=256,  # Shorter for faster training
        MAX_SAMPLES=1000,    # Limit samples for testing

        # Finetuning - LoRA for efficiency
        FINETUNING_TYPE="lora",
        LORA_RANK=8,

        # Training - Small scale for local testing
        BATCH_SIZE=2,
        GRADIENT_ACCUMULATION_STEPS=2,
        LEARNING_RATE=2e-4,
        NUM_EPOCHS=1,  # Just 1 epoch for testing

        # Hardware - Single GPU
        NUM_GPUS=1,
        USE_DEEPSPEED=False,
        USE_BF16=True,

        # Frequent checkpointing for testing
        LOGGING_STEPS=5,
        SAVE_STEPS=50,
        EVAL_STEPS=50,
    )

    # ==================== Prepare Sample Data ====================
    # Check if training data exists
    training_data_path = Path(training_config["INPUT_DATA_PATH"])
    if not training_data_path.exists():
        print(f"\nCreating sample training data at: {training_data_path}")

        import json
        sample_data = [
            {
                "instruction": "What is machine learning?",
                "input": "",
                "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
            },
            {
                "instruction": "Explain the concept of neural networks.",
                "input": "",
                "output": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information using a connectionist approach to computation."
            },
            # Add more samples here...
        ] * 500  # Duplicate to get ~1000 samples

        with open(training_data_path, 'w') as f:
            json.dump(sample_data, f, indent=2)

        print(f"Created {len(sample_data)} sample training examples")

    # ==================== Run Training ====================
    print("\n" + "=" * 80)
    print("Local Training Example")
    print("=" * 80)
    print(f"Model: {training_config['BASE_MODEL']}")
    print(f"Data: {training_config['INPUT_DATA_PATH']}")
    print(f"Output: {output_dir}")
    print(f"Max Samples: {training_config.get('MAX_SAMPLES', 'all')}")
    print("=" * 80)

    results = run_finetuning_workflow(training_config, config)

    # ==================== Results ====================
    print("\n" + "=" * 80)
    print("Local Training Complete!")
    print("=" * 80)
    print(f"Model saved to: {results['model_path']}")
    print(f"\nYou can now use the model:")
    print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("{training_config['BASE_MODEL']}")
model = PeftModel.from_pretrained(model, "{results.get('adapter_path')}")
tokenizer = AutoTokenizer.from_pretrained("{results['model_path']}")
    """)


if __name__ == "__main__":
    main()
