"""
LLM Finetune - Production-ready framework for finetuning large language models.

Main components:
- workflows: Orchestration with Ray
- tasks: Data preparation, training, and packaging
- configs: Configuration management and presets
"""

__version__ = "0.1.0"

from llama_finetune.workflows.finetune_pipeline import (
    run_finetuning_workflow,
    run_evaluation_workflow,
    run_lora_merge_workflow,
)
from llama_finetune.configs.config import (
    Config,
    StorageConfig,
    RayConfig,
    TrainingConfig,
    get_config,
    set_config,
)
from llama_finetune.configs.hyperparameters import (
    HYPERPARAMETERS_LLAMA_BASE,
    HYPERPARAMETERS_LORA,
    HYPERPARAMETERS_QLORA,
    HYPERPARAMETERS_FULL,
    get_config_by_name,
    create_custom_config,
)

__all__ = [
    "__version__",
    # Workflows
    "run_finetuning_workflow",
    "run_evaluation_workflow",
    "run_lora_merge_workflow",
    # Config
    "Config",
    "StorageConfig",
    "RayConfig",
    "TrainingConfig",
    "get_config",
    "set_config",
    # Hyperparameters
    "HYPERPARAMETERS_LLAMA_BASE",
    "HYPERPARAMETERS_LORA",
    "HYPERPARAMETERS_QLORA",
    "HYPERPARAMETERS_FULL",
    "get_config_by_name",
    "create_custom_config",
]
