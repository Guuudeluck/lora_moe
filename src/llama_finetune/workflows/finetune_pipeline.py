"""
LLM Finetuning Pipeline with Ray.

This workflow orchestrates the end-to-end process of finetuning large language models
using transformers + peft + accelerate with Ray for orchestration.

Pipeline stages:
1. Data preparation and formatting
2. Model training with transformers Trainer
3. Model packaging and output

Usage:
    python -m llama_finetune.workflows.finetune_pipeline --config lora
"""

import os
import ray
from typing import Dict, Optional

from llama_finetune.configs.config import get_config, Config
from llama_finetune.configs.hyperparameters import get_config_by_name, HYPERPARAMETERS_LLAMA_BASE
from llama_finetune.tasks.data_preparation import prepare_training_data
from llama_finetune.tasks.training import transformers_trainer
from llama_finetune.tasks.model_packaging import package_model_output, create_model_metadata


def run_finetuning_workflow(
    training_config: Dict,
    config: Optional[Config] = None,
) -> Dict[str, str]:
    """
    Main LLM finetuning workflow using Ray.

    This workflow:
    1. Prepares data in training-compatible format
    2. Executes training with specified configuration
    3. Packages and saves the trained model

    Args:
        training_config: Dict with all hyperparameters (from configs.hyperparameters)
        config: Optional Config object for storage/Ray settings (defaults to env config)

    Returns:
        Dict with workflow results including model path and metrics

    Example:
        from llama_finetune.configs.hyperparameters import HYPERPARAMETERS_LORA
        from llama_finetune.workflows.finetune_pipeline import run_finetuning_workflow

        results = run_finetuning_workflow(HYPERPARAMETERS_LORA)
        print(f"Model saved to: {results['model_path']}")
    """
    if config is None:
        config = get_config()

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            address=config.ray.address,
            runtime_env=config.ray.runtime_env,
            num_cpus=config.ray.num_cpus,
            num_gpus=config.ray.num_gpus,
        )

    print("="*80)
    print("LLM Finetuning Workflow with Ray")
    print("="*80)

    # Extract configuration
    model_id = training_config["MODEL_ID"]
    base_model = training_config["BASE_MODEL"]
    input_data_path = training_config["INPUT_DATA_PATH"]

    # Get output directories from config
    output_base_dir = config.get_output_dir(model_id)
    data_dir = config.get_data_dir(model_id)
    checkpoint_dir = config.get_checkpoint_dir(model_id)

    print(f"\nConfiguration:")
    print(f"  Model ID: {model_id}")
    print(f"  Base Model: {base_model}")
    print(f"  Input Data: {input_data_path}")
    print(f"  Output Directory: {output_base_dir}")
    print(f"  Storage Backend: {config.storage.backend}")

    # ==================== Step 1: Data Preparation ====================
    print(f"\n{'='*80}")
    print("Step 1: Data Preparation")
    print("="*80)

    data_prep_task = prepare_training_data.remote(
        input_data_path=input_data_path,
        output_dir=data_dir,
        dataset_format=training_config.get("DATASET_FORMAT", "alpaca"),
        validation_split=training_config.get("VALIDATION_SPLIT", 0.1),
        max_samples=training_config.get("MAX_SAMPLES"),
    )

    # Wait for data preparation to complete
    data_prep_output = ray.get(data_prep_task)

    print(f"\nData preparation complete:")
    print(f"  Train samples: {data_prep_output['num_train_samples']}")
    print(f"  Validation samples: {data_prep_output['num_val_samples']}")
    print(f"  Train file: {data_prep_output['train_file']}")
    print(f"  Validation file: {data_prep_output['validation_file']}")

    # ==================== Step 2: Model Training ====================
    print(f"\n{'='*80}")
    print("Step 2: Model Training")
    print("="*80)

    training_task = transformers_trainer.remote(
        model_name_or_path=base_model,
        train_file=data_prep_output["train_file"],
        validation_file=data_prep_output["validation_file"],
        output_dir=checkpoint_dir,
        training_config=training_config,
    )

    # Wait for training to complete
    training_output = ray.get(training_task)

    print(f"\nTraining complete:")
    print(f"  Model path: {training_output['model_path']}")
    print(f"  Adapter path: {training_output.get('adapter_dir', 'N/A')}")
    print(f"  Checkpoints: {training_output['num_checkpoints']}")
    print(f"  Train loss: {training_output['metrics'].get('train_loss', 'N/A')}")
    print(f"  Eval loss: {training_output['metrics'].get('eval_loss', 'N/A')}")

    # ==================== Step 3: Model Packaging ====================
    print(f"\n{'='*80}")
    print("Step 3: Model Packaging")
    print("="*80)

    packaging_task = package_model_output.remote(
        model_path=training_output["model_path"],
        output_format=training_config.get("OUTPUT_FORMAT", "huggingface"),
    )

    # Wait for packaging to complete
    final_model_path = ray.get(packaging_task)

    # Create metadata
    metadata_task = create_model_metadata.remote(
        model_path=final_model_path,
        training_config=training_config,
        training_metrics=training_output.get("metrics", {}),
    )

    metadata_path = ray.get(metadata_task)

    print(f"\nPackaging complete:")
    print(f"  Final model: {final_model_path}")
    print(f"  Metadata: {metadata_path}")

    # ==================== Workflow Complete ====================
    print(f"\n{'='*80}")
    print("Workflow Complete!")
    print("="*80)

    results = {
        "status": "success",
        "model_id": model_id,
        "model_path": final_model_path,
        "adapter_path": training_output.get("adapter_dir"),
        "metadata_path": metadata_path,
        "data_dir": data_dir,
        "checkpoint_dir": checkpoint_dir,
        "metrics": training_output.get("metrics", {}),
        "num_train_samples": data_prep_output["num_train_samples"],
        "num_val_samples": data_prep_output["num_val_samples"],
    }

    print(f"\nFinal Results:")
    print(f"  Model ID: {results['model_id']}")
    print(f"  Model Path: {results['model_path']}")
    print(f"  Status: {results['status']}")

    return results


def run_evaluation_workflow(
    model_path: str,
    test_data_path: str,
    config: Optional[Config] = None,
) -> Dict:
    """
    Evaluation-only workflow for existing models.

    Use this workflow to evaluate a pre-trained or finetuned model
    on a test dataset without training.

    Args:
        model_path: Path to model to evaluate
        test_data_path: Path to test data
        config: Optional Config object (defaults to env config)

    Returns:
        Dict with evaluation metrics

    Example:
        from llama_finetune.workflows.finetune_pipeline import run_evaluation_workflow

        results = run_evaluation_workflow(
            model_path="s3://bucket/models/llama-lora-v1",
            test_data_path="s3://bucket/test-data.json"
        )
    """
    if config is None:
        config = get_config()

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            address=config.ray.address,
            runtime_env=config.ray.runtime_env,
        )

    from llama_finetune.tasks.training import evaluate_model

    print("="*80)
    print("Model Evaluation Workflow")
    print("="*80)
    print(f"  Model: {model_path}")
    print(f"  Test Data: {test_data_path}")

    output_dir = config.storage.resolve_path(f"{config.project_name}/evaluation")
    output_file = os.path.join(output_dir, "evaluation_results.json")

    # Run evaluation
    eval_task = evaluate_model.remote(
        model_path=model_path,
        test_file=test_data_path,
        output_file=output_file,
    )

    eval_output = ray.get(eval_task)

    print(f"\nEvaluation complete:")
    print(f"  Results: {output_file}")
    print(f"  Eval loss: {eval_output.get('eval_loss', 'N/A')}")

    return eval_output


def run_lora_merge_workflow(
    base_model_path: str,
    lora_adapter_path: str,
    output_dir: Optional[str] = None,
    config: Optional[Config] = None,
) -> str:
    """
    Merge LoRA adapter weights with base model.

    Use this workflow to merge a trained LoRA adapter back into
    the base model for easier deployment.

    Args:
        base_model_path: Path to base model
        lora_adapter_path: Path to LoRA adapter
        output_dir: Optional output directory (defaults to config-based path)
        config: Optional Config object (defaults to env config)

    Returns:
        Path to merged model

    Example:
        from llama_finetune.workflows.finetune_pipeline import run_lora_merge_workflow

        merged_path = run_lora_merge_workflow(
            base_model_path="meta-llama/Llama-2-7b-hf",
            lora_adapter_path="s3://bucket/models/llama-lora-v1/adapter"
        )
    """
    if config is None:
        config = get_config()

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            address=config.ray.address,
            runtime_env=config.ray.runtime_env,
        )

    from llama_finetune.tasks.model_packaging import merge_lora_weights

    if output_dir is None:
        output_dir = config.storage.resolve_path(f"{config.project_name}/merged_model")

    print("="*80)
    print("LoRA Merge Workflow")
    print("="*80)
    print(f"  Base Model: {base_model_path}")
    print(f"  LoRA Adapter: {lora_adapter_path}")
    print(f"  Output: {output_dir}")

    # Merge weights
    merge_task = merge_lora_weights.remote(
        base_model_path=base_model_path,
        lora_adapter_path=lora_adapter_path,
        output_dir=output_dir,
    )

    merged_model_path = ray.get(merge_task)

    print(f"\nLoRA merge complete: {merged_model_path}")

    return merged_model_path


# ==================== CLI Entry Point ====================

def main():
    """CLI entry point for running workflows."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Finetuning with Ray")
    parser.add_argument(
        "--workflow",
        type=str,
        choices=["finetune", "evaluate", "merge"],
        default="finetune",
        help="Workflow to run (default: finetune)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="base",
        help="Configuration preset name (default: base). Options: base, lora, qlora, full, large, instruction, chat"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Model path (for evaluate/merge workflows)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Test data path (for evaluate workflow)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Base model path (for merge workflow)"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="LoRA adapter path (for merge workflow)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (optional)"
    )

    args = parser.parse_args()

    # Load configuration from environment
    config = get_config()

    if args.workflow == "finetune":
        # Load training configuration
        training_config = get_config_by_name(args.config)

        # Run finetuning workflow
        results = run_finetuning_workflow(training_config, config)

        print(f"\n✓ Finetuning complete!")
        print(f"  Model: {results['model_path']}")

    elif args.workflow == "evaluate":
        if not args.model_path or not args.test_data:
            parser.error("--model-path and --test-data are required for evaluate workflow")

        # Run evaluation workflow
        results = run_evaluation_workflow(args.model_path, args.test_data, config)

        print(f"\n✓ Evaluation complete!")
        print(f"  Loss: {results.get('eval_loss', 'N/A')}")

    elif args.workflow == "merge":
        if not args.base_model or not args.adapter_path:
            parser.error("--base-model and --adapter-path are required for merge workflow")

        # Run merge workflow
        merged_path = run_lora_merge_workflow(
            args.base_model,
            args.adapter_path,
            args.output_dir,
            config
        )

        print(f"\n✓ Merge complete!")
        print(f"  Merged model: {merged_path}")


if __name__ == "__main__":
    main()
