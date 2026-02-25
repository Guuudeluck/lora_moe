"""
Configuration management for LLM finetuning.

Replaces LinkedIn's per-user settings with environment-based configuration
that supports S3, local filesystem, and cloud storage.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class StorageConfig:
    """Storage configuration for data and models."""

    # Storage backend: 's3', 'local', 'azure', or 'gcs'
    backend: str = field(default_factory=lambda: os.getenv("STORAGE_BACKEND", "local"))

    # Base path for data and models (can be S3 URI like s3://bucket/path or local path)
    base_path: str = field(default_factory=lambda: os.getenv("STORAGE_BASE_PATH", "./outputs"))

    # AWS S3 specific settings
    aws_access_key_id: Optional[str] = field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key: Optional[str] = field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY"))
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    s3_endpoint_url: Optional[str] = field(default_factory=lambda: os.getenv("S3_ENDPOINT_URL"))

    # Azure Blob Storage specific settings
    azure_account_name: Optional[str] = field(default_factory=lambda: os.getenv("AZURE_ACCOUNT_NAME"))
    azure_account_key: Optional[str] = field(default_factory=lambda: os.getenv("AZURE_ACCOUNT_KEY"))

    # Google Cloud Storage specific settings
    gcs_project: Optional[str] = field(default_factory=lambda: os.getenv("GCS_PROJECT"))
    gcs_credentials_path: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

    def get_fsspec_options(self) -> Dict[str, Any]:
        """Get fsspec storage options based on backend."""
        if self.backend == "s3":
            options = {
                "key": self.aws_access_key_id,
                "secret": self.aws_secret_access_key,
                "client_kwargs": {"region_name": self.aws_region}
            }
            if self.s3_endpoint_url:
                options["client_kwargs"]["endpoint_url"] = self.s3_endpoint_url
            return options
        elif self.backend == "azure":
            return {
                "account_name": self.azure_account_name,
                "account_key": self.azure_account_key,
            }
        elif self.backend == "gcs":
            return {
                "project": self.gcs_project,
                "token": self.gcs_credentials_path,
            }
        return {}

    def resolve_path(self, relative_path: str) -> str:
        """Resolve a relative path against the base path."""
        if self.backend == "local":
            return str(Path(self.base_path) / relative_path)
        else:
            # For cloud storage, join paths with forward slashes
            base = self.base_path.rstrip("/")
            rel = relative_path.lstrip("/")
            return f"{base}/{rel}"


@dataclass
class RayConfig:
    """Ray cluster configuration."""

    # Ray cluster address (None for local mode)
    address: Optional[str] = field(default_factory=lambda: os.getenv("RAY_ADDRESS"))

    # Ray runtime environment
    runtime_env: Optional[Dict[str, Any]] = None

    # Number of GPUs to use (None = auto-detect)
    num_gpus: Optional[int] = field(default_factory=lambda:
        int(os.getenv("NUM_GPUS")) if os.getenv("NUM_GPUS") else None
    )

    # Number of CPUs to use (None = auto-detect)
    num_cpus: Optional[int] = field(default_factory=lambda:
        int(os.getenv("NUM_CPUS")) if os.getenv("NUM_CPUS") else None
    )


@dataclass
class TrainingConfig:
    """Training execution configuration."""

    # Docker image for training (if using Ray on Kubernetes)
    container_image: str = field(default_factory=lambda:
        os.getenv("TRAINING_CONTAINER_IMAGE", "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04")
    )

    # Working directory for training
    working_dir: str = field(default_factory=lambda: os.getenv("WORKING_DIR", "/workspace"))

    # Enable wandb logging
    enable_wandb: bool = field(default_factory=lambda: os.getenv("ENABLE_WANDB", "false").lower() == "true")
    wandb_project: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_PROJECT"))
    wandb_entity: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_ENTITY"))

    # Enable tensorboard logging
    enable_tensorboard: bool = field(default_factory=lambda:
        os.getenv("ENABLE_TENSORBOARD", "true").lower() == "true"
    )

    # Enable MLflow logging
    enable_mlflow: bool = field(default_factory=lambda: os.getenv("ENABLE_MLFLOW", "false").lower() == "true")
    mlflow_tracking_uri: Optional[str] = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI"))


@dataclass
class Config:
    """Main configuration for LLM finetuning."""

    # Sub-configurations
    storage: StorageConfig = field(default_factory=StorageConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Project name
    project_name: str = field(default_factory=lambda: os.getenv("PROJECT_NAME", "llm-finetune"))

    # Experiment name
    experiment_name: str = field(default_factory=lambda: os.getenv("EXPERIMENT_NAME", "default"))

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        storage = StorageConfig(**config_dict.get("storage", {}))
        ray = RayConfig(**config_dict.get("ray", {}))
        training = TrainingConfig(**config_dict.get("training", {}))

        return cls(
            storage=storage,
            ray=ray,
            training=training,
            project_name=config_dict.get("project_name", "llm-finetune"),
            experiment_name=config_dict.get("experiment_name", "default"),
        )

    def get_output_dir(self, model_id: str) -> str:
        """Get output directory for a specific model."""
        path = f"{self.project_name}/{self.experiment_name}/{model_id}"
        return self.storage.resolve_path(path)

    def get_data_dir(self, model_id: str) -> str:
        """Get data directory for a specific model."""
        return f"{self.get_output_dir(model_id)}/data"

    def get_checkpoint_dir(self, model_id: str) -> str:
        """Get checkpoint directory for a specific model."""
        return f"{self.get_output_dir(model_id)}/checkpoints"


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config
    _config = None
