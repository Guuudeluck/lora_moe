"""
Tests for configuration management.
"""

import os
import pytest
from llama_finetune.configs.config import (
    Config,
    StorageConfig,
    RayConfig,
    TrainingConfig,
    get_config,
    set_config,
    reset_config,
)
from llama_finetune.configs.hyperparameters import (
    get_config_by_name,
    create_custom_config,
    HYPERPARAMETERS_LORA,
)


def test_storage_config_defaults():
    """Test StorageConfig with default values."""
    config = StorageConfig()
    assert config.backend == "local"
    assert config.base_path == "./outputs"


def test_storage_config_s3():
    """Test StorageConfig for S3."""
    config = StorageConfig(
        backend="s3",
        base_path="s3://bucket/path",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
        aws_region="us-west-2",
    )
    assert config.backend == "s3"
    assert config.base_path == "s3://bucket/path"

    options = config.get_fsspec_options()
    assert options["key"] == "key"
    assert options["secret"] == "secret"


def test_storage_config_resolve_path():
    """Test path resolution."""
    config = StorageConfig(backend="local", base_path="/home/user/outputs")
    path = config.resolve_path("models/model1")
    assert path == "/home/user/outputs/models/model1"


def test_ray_config_defaults():
    """Test RayConfig with default values."""
    config = RayConfig()
    assert config.address is None
    assert config.num_gpus is None


def test_config_defaults():
    """Test Config with default values."""
    reset_config()
    config = get_config()
    assert isinstance(config, Config)
    assert isinstance(config.storage, StorageConfig)
    assert isinstance(config.ray, RayConfig)
    assert config.project_name == "llm-finetune"


def test_config_set_and_get():
    """Test setting and getting global config."""
    custom_config = Config(
        project_name="test-project",
        experiment_name="test-experiment",
    )
    set_config(custom_config)

    retrieved_config = get_config()
    assert retrieved_config.project_name == "test-project"
    assert retrieved_config.experiment_name == "test-experiment"

    reset_config()


def test_get_config_by_name():
    """Test getting hyperparameter config by name."""
    config = get_config_by_name("lora")
    assert config["FINETUNING_TYPE"] == "lora"
    assert config["MODEL_ID"] == "llama-lora-v1"


def test_get_config_by_name_invalid():
    """Test getting config with invalid name."""
    with pytest.raises(ValueError):
        get_config_by_name("invalid_name")


def test_create_custom_config():
    """Test creating custom configuration."""
    config = create_custom_config(
        MODEL_ID="custom-model",
        BASE_MODEL="mistralai/Mistral-7B-v0.1",
        LEARNING_RATE=1e-4,
    )
    assert config["MODEL_ID"] == "custom-model"
    assert config["BASE_MODEL"] == "mistralai/Mistral-7B-v0.1"
    assert config["LEARNING_RATE"] == 1e-4
    # Check that base config values are inherited
    assert "FINETUNING_TYPE" in config


def test_config_get_output_dir():
    """Test output directory generation."""
    config = Config(
        storage=StorageConfig(backend="local", base_path="/outputs"),
        project_name="test-project",
        experiment_name="exp1",
    )
    output_dir = config.get_output_dir("model1")
    assert output_dir == "/outputs/test-project/exp1/model1"


def test_config_from_dict():
    """Test creating Config from dictionary."""
    config_dict = {
        "storage": {
            "backend": "s3",
            "base_path": "s3://bucket/path",
        },
        "project_name": "my-project",
        "experiment_name": "exp1",
    }
    config = Config.from_dict(config_dict)
    assert config.storage.backend == "s3"
    assert config.storage.base_path == "s3://bucket/path"
    assert config.project_name == "my-project"
