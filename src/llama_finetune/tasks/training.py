"""
Training task for LLM finetuning using transformers + peft + accelerate with Ray.

This module executes LLM finetuning with support for:
- LoRA, QLoRA, and full finetuning via PEFT
- Distributed training with DeepSpeed
- Multi-GPU training with Accelerate
- Direct transformers.Trainer API
"""

import ray
from typing import Dict, Optional
import os


@ray.remote(num_cpus=30, num_gpus=8, memory=360 * 1024 * 1024 * 1024)  # 30 CPUs, 8 GPUs, 360Gi
def transformers_trainer(
    model_name_or_path: str,
    train_file: str,
    validation_file: str,
    output_dir: str,
    training_config: Dict,
) -> Dict[str, str]:
    """
    Execute LLM training using transformers + peft + accelerate.

    Args:
        model_name_or_path: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf") or local path
        train_file: Path to training data (JSON/JSONL)
        validation_file: Path to validation data (JSON/JSONL)
        output_dir: Directory to save checkpoints and outputs
        training_config: Dict with all training arguments:
            - finetuning_type: "lora", "qlora", or "full"
            - lora_rank: LoRA rank (default: 8)
            - lora_alpha: LoRA alpha (default: 16)
            - lora_dropout: LoRA dropout (default: 0.05)
            - lora_target_modules: List of target modules
            - batch_size: Per-device batch size
            - gradient_accumulation_steps: Gradient accumulation steps
            - learning_rate: Learning rate
            - num_epochs: Number of training epochs
            - max_seq_length: Maximum sequence length
            - num_gpus: Number of GPUs to use
            - use_deepspeed: Whether to use DeepSpeed
            - deepspeed_stage: DeepSpeed ZeRO stage (2 or 3)
            - save_steps: Save checkpoint every N steps
            - eval_steps: Evaluate every N steps
            - logging_steps: Log every N steps

    Returns:
        Dict with model_path, metrics, logs, and training info
    """
    import os
    import json
    import torch
    from pathlib import Path
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from datasets import load_dataset
    import transformers

    print("="*80)
    print("LLM Training with transformers + peft + accelerate")
    print("="*80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract training configuration
    finetuning_type = training_config.get("finetuning_type", "lora").lower()
    num_gpus = training_config.get("num_gpus", 8)
    max_seq_length = training_config.get("max_seq_length", 512)

    print(f"\nConfiguration:")
    print(f"  Model: {model_name_or_path}")
    print(f"  Finetuning type: {finetuning_type}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Max sequence length: {max_seq_length}")

    # ==================== Load Tokenizer ====================
    print(f"\n{'='*80}")
    print("Loading tokenizer...")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Padding token: {tokenizer.pad_token}")

    # ==================== Load Model ====================
    print(f"\n{'='*80}")
    print("Loading model...")
    print("="*80)

    # Determine quantization config for QLoRA
    quantization_config = None
    if finetuning_type == "qlora":
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if training_config.get("use_bf16", True) else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization (QLoRA)")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if training_config.get("use_bf16", True) else torch.float16,
        device_map="auto" if num_gpus > 1 else None,
        trust_remote_code=True,
    )

    print(f"Model loaded: {model.__class__.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ==================== Apply PEFT (LoRA/QLoRA) ====================
    if finetuning_type in ["lora", "qlora"]:
        print(f"\n{'='*80}")
        print(f"Applying {finetuning_type.upper()} configuration...")
        print("="*80)

        # Prepare model for k-bit training (for QLoRA)
        if finetuning_type == "qlora":
            model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=training_config.get("lora_rank", 8),
            lora_alpha=training_config.get("lora_alpha", 16),
            lora_dropout=training_config.get("lora_dropout", 0.05),
            target_modules=training_config.get(
                "lora_target_modules",
                ["q_proj", "v_proj", "k_proj", "o_proj"]
            ),
            bias="none",
        )

        print(f"LoRA Configuration:")
        print(f"  Rank (r): {lora_config.r}")
        print(f"  Alpha: {lora_config.lora_alpha}")
        print(f"  Dropout: {lora_config.lora_dropout}")
        print(f"  Target modules: {lora_config.target_modules}")

        # Apply PEFT
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing if requested
    if training_config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # ==================== Load and Prepare Dataset ====================
    print(f"\n{'='*80}")
    print("Loading datasets...")
    print("="*80)

    # Load datasets
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": validation_file,
        }
    )

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")

    # Tokenize function
    def tokenize_function(examples):
        """
        Tokenize examples for causal language modeling.

        Assumes data format:
        - alpaca: {"instruction": "...", "input": "...", "output": "..."}
        - sharegpt: {"conversations": [...]}
        """
        texts = []

        for i in range(len(examples.get("instruction", []))):
            # Format text based on dataset format
            if "instruction" in examples:
                # Alpaca format
                instruction = examples["instruction"][i]
                input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
                output = examples["output"][i]

                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            else:
                # Fallback to raw text if available
                text = examples.get("text", [""])[i]

            texts.append(text)

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )

        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    print("Tokenization complete")

    # ==================== Configure Training Arguments ====================
    print(f"\n{'='*80}")
    print("Configuring training arguments...")
    print("="*80)

    # Determine if using DeepSpeed
    deepspeed_config = None
    if training_config.get("use_deepspeed", False) and num_gpus > 1:
        deepspeed_stage = training_config.get("deepspeed_stage", 2)
        deepspeed_config_path = os.path.join(output_dir, f"ds_config_zero{deepspeed_stage}.json")

        # Create DeepSpeed config
        ds_config = {
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "fp16": {
                "enabled": not training_config.get("use_bf16", True)
            },
            "bf16": {
                "enabled": training_config.get("use_bf16", True)
            },
            "zero_optimization": {
                "stage": deepspeed_stage,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                } if deepspeed_stage == 3 else None,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                } if deepspeed_stage == 3 else None,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e7,
                "stage3_prefetch_bucket_size": 5e7 if deepspeed_stage == 3 else None,
                "stage3_param_persistence_threshold": 1e5 if deepspeed_stage == 3 else None,
            }
        }

        with open(deepspeed_config_path, 'w') as f:
            json.dump(ds_config, f, indent=2)

        deepspeed_config = deepspeed_config_path
        print(f"DeepSpeed ZeRO-{deepspeed_stage} config saved to: {deepspeed_config_path}")

    # Create TrainingArguments
    training_args = TrainingArguments(
        # Output
        output_dir=output_dir,
        overwrite_output_dir=False,

        # Training hyperparameters
        num_train_epochs=training_config.get("num_epochs", 3),
        per_device_train_batch_size=training_config.get("batch_size", 4),
        per_device_eval_batch_size=training_config.get("batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 5e-5),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),

        # Optimization
        optim="adamw_torch",
        weight_decay=0.0,

        # Mixed precision
        fp16=not training_config.get("use_bf16", True),
        bf16=training_config.get("use_bf16", True),

        # Evaluation
        evaluation_strategy="steps",
        eval_steps=training_config.get("eval_steps", 1000),
        do_eval=True,

        # Logging
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=training_config.get("logging_steps", 10),
        report_to=["tensorboard"],

        # Checkpointing
        save_strategy="steps",
        save_steps=training_config.get("save_steps", 1000),
        save_total_limit=training_config.get("save_total_limit", 3),

        # DeepSpeed
        deepspeed=deepspeed_config,

        # Other
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    print("Training arguments configured:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Mixed precision: {'bf16' if training_args.bf16 else 'fp16' if training_args.fp16 else 'fp32'}")

    # ==================== Create Trainer ====================
    print(f"\n{'='*80}")
    print("Creating Trainer...")
    print("="*80)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    print("Trainer created")

    # ==================== Train Model ====================
    print(f"\n{'='*80}")
    print("Starting training...")
    print("="*80 + "\n")

    try:
        # Train
        train_result = trainer.train()

        print("\n" + "="*80)
        print("Training completed successfully!")
        print("="*80)

        # Save final metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise

    # ==================== Save Model ====================
    print(f"\n{'='*80}")
    print("Saving model...")
    print("="*80)

    # Save model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # For PEFT models, also save the full adapter
    if finetuning_type in ["lora", "qlora"]:
        adapter_dir = os.path.join(output_dir, "adapter")
        model.save_pretrained(adapter_dir)
        print(f"LoRA adapter saved to: {adapter_dir}")

    print(f"Model saved to: {output_dir}")

    # ==================== Evaluate and Collect Metrics ====================
    print(f"\n{'='*80}")
    print("Running final evaluation...")
    print("="*80)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print(f"Final evaluation loss: {eval_metrics.get('eval_loss', 'N/A')}")

    # Collect all metrics
    all_metrics = {
        "status": "completed",
        "train_loss": metrics.get("train_loss"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "train_runtime": metrics.get("train_runtime"),
        "train_samples_per_second": metrics.get("train_samples_per_second"),
    }

    # Find checkpoints
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))

    best_checkpoint = output_dir  # Trainer loads best model at end

    print(f"\n{'='*80}")
    print("Training Summary")
    print("="*80)
    print(f"Best model: {best_checkpoint}")
    print(f"Checkpoints: {len(checkpoint_dirs)}")
    print(f"Final train loss: {all_metrics['train_loss']:.4f}" if all_metrics['train_loss'] else "")
    print(f"Final eval loss: {all_metrics['eval_loss']:.4f}" if all_metrics['eval_loss'] else "")
    print("="*80 + "\n")

    return {
        "model_path": best_checkpoint,
        "output_dir": output_dir,
        "adapter_dir": os.path.join(output_dir, "adapter") if finetuning_type in ["lora", "qlora"] else None,
        "metrics": all_metrics,
        "num_checkpoints": len(checkpoint_dirs),
    }


@ray.remote(num_cpus=4, num_gpus=1, memory=32 * 1024 * 1024 * 1024)  # 4 CPUs, 1 GPU, 32Gi
def evaluate_model(
    model_path: str,
    test_file: str,
    output_file: str,
) -> Dict[str, any]:
    """
    Evaluate trained model on test set.

    Args:
        model_path: Path to trained model checkpoint
        test_file: Path to test data
        output_file: Path to save evaluation results

    Returns:
        Dict with evaluation metrics
    """
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from datasets import load_dataset
    from peft import PeftModel

    print(f"Evaluating model at: {model_path}")
    print(f"Test data: {test_file}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (check if it's a PEFT model)
    try:
        # Try loading as PEFT model
        from peft import PeftConfig
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    except:
        # Load as regular model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # Load test dataset
    test_dataset = load_dataset("json", data_files={"test": test_file})["test"]

    # Tokenize (simple version)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"] if "text" in examples else examples["instruction"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Create trainer for evaluation
    training_args = TrainingArguments(
        output_dir="./eval_temp",
        per_device_eval_batch_size=4,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
    )

    # Evaluate
    results = trainer.evaluate()

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation results saved to: {output_file}")
    print(f"Eval loss: {results.get('eval_loss', 'N/A')}")

    return results
