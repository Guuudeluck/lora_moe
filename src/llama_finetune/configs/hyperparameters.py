"""
Hyperparameter configurations for LLM finetuning with transformers + peft + accelerate.

This file provides default hyperparameters and configuration templates
that can be customized for specific use cases.

Configuration keys map directly to transformers.TrainingArguments and peft.LoraConfig:
- batch_size -> per_device_train_batch_size
- num_epochs -> num_train_epochs
- learning_rate -> learning_rate
- lora_rank -> LoraConfig.r
- lora_alpha -> LoraConfig.lora_alpha
- lora_dropout -> LoraConfig.lora_dropout
- lora_target_modules -> LoraConfig.target_modules
"""

# Base hyperparameters for LLaMA finetuning
HYPERPARAMETERS_LLAMA_BASE = {
    # ==================== Experiment Identification ====================
    "MODEL_ID": "llama-finetune-v1",
    "BASE_MODEL": "meta-llama/Llama-2-7b-hf",  # HuggingFace model ID
    "MODEL_DESCRIPTION": "General purpose LLaMA-2 finetuning",

    # ==================== Data Configuration ====================
    "INPUT_DATA_PATH": "s3://your-bucket/training-data/data.json",  # S3 path or local path
    "DATASET_FORMAT": "alpaca",  # alpaca, sharegpt, or custom
    "MAX_SEQ_LENGTH": 512,
    "VALIDATION_SPLIT": 0.1,  # Fraction of data for validation
    "MAX_SAMPLES": None,  # None = use all data

    # ==================== Finetuning Type ====================
    "FINETUNING_TYPE": "lora",  # lora, qlora, full

    # LoRA Configuration
    "LORA_RANK": 8,
    "LORA_ALPHA": 16,
    "LORA_DROPOUT": 0.05,
    "LORA_TARGET_MODULES": ["q_proj", "v_proj", "k_proj", "o_proj"],

    # ==================== Training Hyperparameters ====================
    "BATCH_SIZE": 4,  # Per-device batch size
    "GRADIENT_ACCUMULATION_STEPS": 4,  # Effective batch size = batch_size * grad_accum * num_gpus
    "LEARNING_RATE": 5e-5,
    "NUM_EPOCHS": 3,
    "WARMUP_RATIO": 0.1,
    "LR_SCHEDULER_TYPE": "cosine",  # cosine, linear, constant
    "MAX_GRAD_NORM": 1.0,

    # ==================== GPU Configuration ====================
    "NUM_GPUS": 8,
    "GPU_TYPE": "h100_8",  # h100_8, a100_8, etc. (for documentation)
    "TRAINER_MEMORY": 360,  # GB (for documentation)

    # ==================== Distributed Training ====================
    "USE_DEEPSPEED": False,  # Use DeepSpeed for distributed training
    "DEEPSPEED_STAGE": 2,  # ZeRO stage (2 or 3)

    # ==================== Optimization ====================
    "USE_BF16": True,  # Use bfloat16 (recommended for H100/A100)
    "GRADIENT_CHECKPOINTING": False,  # Save memory at cost of speed

    # ==================== Logging and Checkpointing ====================
    "LOGGING_STEPS": 10,
    "SAVE_STEPS": 1000,
    "EVAL_STEPS": 1000,
    "SAVE_TOTAL_LIMIT": 3,  # Keep only last N checkpoints

    # ==================== Output Configuration ====================
    "OUTPUT_FORMAT": "huggingface",  # huggingface, onnx, quantized
}


# ==================== Preset Configurations ====================

# Configuration for LoRA finetuning (memory efficient)
HYPERPARAMETERS_LORA = {
    **HYPERPARAMETERS_LLAMA_BASE,
    "MODEL_ID": "llama-lora-v1",
    "FINETUNING_TYPE": "lora",
    "LORA_RANK": 8,
    "BATCH_SIZE": 4,
    "NUM_GPUS": 4,
    "USE_DEEPSPEED": False,
}

# Configuration for QLoRA finetuning (most memory efficient)
HYPERPARAMETERS_QLORA = {
    **HYPERPARAMETERS_LLAMA_BASE,
    "MODEL_ID": "llama-qlora-v1",
    "FINETUNING_TYPE": "qlora",
    "LORA_RANK": 8,
    "BATCH_SIZE": 2,
    "NUM_GPUS": 2,
    "USE_DEEPSPEED": False,
}

# Configuration for full finetuning (best quality, most resources)
HYPERPARAMETERS_FULL = {
    **HYPERPARAMETERS_LLAMA_BASE,
    "MODEL_ID": "llama-full-v1",
    "FINETUNING_TYPE": "full",
    "BATCH_SIZE": 2,
    "NUM_GPUS": 8,
    "USE_DEEPSPEED": True,
    "DEEPSPEED_STAGE": 3,
    "GRADIENT_CHECKPOINTING": True,
}

# Configuration for large models (13B+)
HYPERPARAMETERS_LARGE_MODEL = {
    **HYPERPARAMETERS_LLAMA_BASE,
    "MODEL_ID": "llama-13b-lora-v1",
    "BASE_MODEL": "meta-llama/Llama-2-13b-hf",
    "FINETUNING_TYPE": "lora",
    "LORA_RANK": 16,
    "BATCH_SIZE": 2,
    "GRADIENT_ACCUMULATION_STEPS": 8,
    "NUM_GPUS": 8,
    "USE_DEEPSPEED": True,
    "DEEPSPEED_STAGE": 2,
    "GRADIENT_CHECKPOINTING": True,
}

# Configuration for instruction tuning
HYPERPARAMETERS_INSTRUCTION = {
    **HYPERPARAMETERS_LLAMA_BASE,
    "MODEL_ID": "llama-instruction-v1",
    "DATASET_FORMAT": "alpaca",
    "MAX_SEQ_LENGTH": 512,
    "NUM_EPOCHS": 3,
}

# Configuration for chat model finetuning
HYPERPARAMETERS_CHAT = {
    **HYPERPARAMETERS_LLAMA_BASE,
    "MODEL_ID": "llama-chat-v1",
    "BASE_MODEL": "meta-llama/Llama-2-7b-chat-hf",
    "DATASET_FORMAT": "sharegpt",
    "MAX_SEQ_LENGTH": 1024,
    "NUM_EPOCHS": 1,
}


def get_config_by_name(config_name: str) -> dict:
    """
    Get configuration by name.

    Args:
        config_name: Name of configuration preset

    Returns:
        Configuration dictionary

    Available presets:
        - "base": Base configuration
        - "lora": LoRA finetuning
        - "qlora": QLoRA finetuning
        - "full": Full finetuning
        - "large": Large model (13B+)
        - "instruction": Instruction tuning
        - "chat": Chat model finetuning
    """
    configs = {
        "base": HYPERPARAMETERS_LLAMA_BASE,
        "lora": HYPERPARAMETERS_LORA,
        "qlora": HYPERPARAMETERS_QLORA,
        "full": HYPERPARAMETERS_FULL,
        "large": HYPERPARAMETERS_LARGE_MODEL,
        "instruction": HYPERPARAMETERS_INSTRUCTION,
        "chat": HYPERPARAMETERS_CHAT,
    }

    if config_name not in configs:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available: {list(configs.keys())}"
        )

    return configs[config_name]


def create_custom_config(**kwargs) -> dict:
    """
    Create a custom configuration by overriding base config.

    Example:
        config = create_custom_config(
            MODEL_ID="my-custom-model",
            BASE_MODEL="mistralai/Mistral-7B-v0.1",
            LEARNING_RATE=1e-4,
        )

    Returns:
        Custom configuration dictionary
    """
    config = HYPERPARAMETERS_LLAMA_BASE.copy()
    config.update(kwargs)
    return config
