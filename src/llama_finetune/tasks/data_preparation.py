"""
Data preparation task for LLM finetuning with Ray.

This module handles conversion of various data formats to training-compatible formats.
Supports: Alpaca, ShareGPT, and custom formats.
"""

import ray
from typing import Dict, Optional
import os


@ray.remote(num_cpus=4, memory=32 * 1024 * 1024 * 1024)  # 32Gi
def prepare_training_data(
    input_data_path: str,
    output_dir: str,
    dataset_format: str = "alpaca",
    validation_split: float = 0.1,
    max_samples: Optional[int] = None,
) -> Dict[str, str]:
    """
    Prepare training data for LLM finetuning.

    Supports multiple formats:
    - alpaca: {"instruction": "...", "input": "...", "output": "..."}
    - sharegpt: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    - custom: User-defined format (requires custom parsing logic)

    Args:
        input_data_path: Path to input data (JSON/JSONL/CSV). Supports S3, local paths.
        output_dir: Directory to save formatted data
        dataset_format: Format specification (alpaca, sharegpt, custom)
        validation_split: Fraction of data to use for validation (default: 0.1)
        max_samples: Maximum number of samples to use (None = use all)

    Returns:
        Dict with paths to train/validation files and dataset info
    """
    import json
    import os
    from pathlib import Path
    import random

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read input data based on file extension and path type
    print(f"Reading data from: {input_data_path}")

    # Handle different path types (S3, local)
    if input_data_path.startswith("s3://"):
        import fsspec
        # Get S3 credentials from environment or config
        storage_options = {}
        if os.getenv("AWS_ACCESS_KEY_ID"):
            storage_options["key"] = os.getenv("AWS_ACCESS_KEY_ID")
            storage_options["secret"] = os.getenv("AWS_SECRET_ACCESS_KEY")
        if os.getenv("AWS_REGION"):
            storage_options["client_kwargs"] = {"region_name": os.getenv("AWS_REGION")}

        fs = fsspec.filesystem("s3", **storage_options)
        with fs.open(input_data_path, 'r') as f:
            if input_data_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
    else:
        # Local file
        with open(input_data_path, 'r', encoding='utf-8') as f:
            if input_data_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)

    print(f"Loaded {len(data)} samples from input")

    # Limit samples if specified
    if max_samples and max_samples < len(data):
        random.seed(42)
        data = random.sample(data, max_samples)
        print(f"Sampled {max_samples} samples")

    # Validate and convert data format
    print(f"Processing data in {dataset_format} format")

    if dataset_format == "alpaca":
        # Validate alpaca format
        for i, item in enumerate(data[:5]):  # Check first 5 samples
            required_keys = ["instruction", "output"]
            if not all(k in item for k in required_keys):
                print(f"Warning: Sample {i} missing required keys: {required_keys}")

    elif dataset_format == "sharegpt":
        # Validate sharegpt format
        for i, item in enumerate(data[:5]):
            if "conversations" not in item:
                print(f"Warning: Sample {i} missing 'conversations' key")

    else:
        print(f"Using custom format: {dataset_format}")
        # Add custom format handling logic here if needed

    # Split into train/validation
    random.seed(42)
    random.shuffle(data)

    val_size = int(len(data) * validation_split)
    train_data = data[val_size:]
    val_data = data[:val_size]

    print(f"Split: {len(train_data)} training samples, {len(val_data)} validation samples")

    # Save train data
    train_file = os.path.join(output_dir, "train.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"Saved training data to: {train_file}")

    # Save validation data
    validation_file = os.path.join(output_dir, "validation.json")
    with open(validation_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"Saved validation data to: {validation_file}")

    # Create dataset info summary
    dataset_info = {
        "format": dataset_format,
        "num_train_samples": len(train_data),
        "num_val_samples": len(val_data),
        "train_file": "train.json",
        "validation_file": "validation.json",
    }

    dataset_info_file = os.path.join(output_dir, "dataset_info.json")
    with open(dataset_info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"Saved dataset info to: {dataset_info_file}")

    return {
        "train_file": train_file,
        "validation_file": validation_file,
        "dataset_info": dataset_info_file,
        "num_train_samples": len(train_data),
        "num_val_samples": len(val_data),
    }


@ray.remote(num_cpus=2, memory=16 * 1024 * 1024 * 1024)  # 16Gi
def validate_data_format(
    data_file: str,
    expected_format: str,
) -> Dict[str, any]:
    """
    Validate data format and provide statistics.

    Args:
        data_file: Path to data file to validate
        expected_format: Expected format (alpaca, sharegpt, custom)

    Returns:
        Dict with validation results and statistics
    """
    import json

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {
        "num_samples": len(data),
        "is_valid": True,
        "errors": [],
    }

    if expected_format == "alpaca":
        for i, item in enumerate(data):
            if "instruction" not in item or "output" not in item:
                stats["is_valid"] = False
                stats["errors"].append(f"Sample {i}: Missing required keys")

    elif expected_format == "sharegpt":
        for i, item in enumerate(data):
            if "conversations" not in item:
                stats["is_valid"] = False
                stats["errors"].append(f"Sample {i}: Missing 'conversations' key")

    return stats
