"""
Model packaging task for trained LLMs with Ray.

This module handles packaging and exporting trained models to various formats:
- HuggingFace format (default)
- ONNX format
- Quantized models (INT8, INT4)
"""

import ray
from typing import Dict, Optional
import os


@ray.remote(num_cpus=4, memory=16 * 1024 * 1024 * 1024)  # 16Gi
def package_model_output(
    model_path: str,
    output_format: str = "huggingface",
    output_dir: Optional[str] = None,
) -> str:
    """
    Package trained model for deployment.

    Args:
        model_path: Path to trained model checkpoint
        output_format: Output format:
            - "huggingface": Keep as HuggingFace format (default)
            - "onnx": Convert to ONNX format
            - "quantized": Quantize model (INT8)
        output_dir: Optional output directory (default: same as model_path)

    Returns:
        Path to packaged model
    """
    import os
    import shutil
    from pathlib import Path

    print(f"Packaging model from: {model_path}")
    print(f"Output format: {output_format}")

    if output_dir is None:
        output_dir = model_path

    os.makedirs(output_dir, exist_ok=True)

    if output_format == "huggingface":
        # Model is already in HuggingFace format
        print("Model is already in HuggingFace format")

        # Create model card
        model_card_path = os.path.join(output_dir, "MODEL_CARD.md")
        with open(model_card_path, 'w') as f:
            f.write("# Finetuned Language Model\n\n")
            f.write(f"## Model Details\n\n")
            f.write(f"- Model Path: {model_path}\n")
            f.write(f"- Format: HuggingFace Transformers\n")
            f.write(f"\n## Usage\n\n")
            f.write("```python\n")
            f.write("from transformers import AutoModelForCausalLM, AutoTokenizer\n\n")
            f.write(f"model = AutoModelForCausalLM.from_pretrained('{output_dir}')\n")
            f.write(f"tokenizer = AutoTokenizer.from_pretrained('{output_dir}')\n")
            f.write("```\n")

        print(f"Created model card: {model_card_path}")

    elif output_format == "onnx":
        print("Converting model to ONNX format...")
        # TODO: Implement ONNX conversion
        # This would require transformers.onnx or optimum library
        print("ONNX conversion not yet implemented")
        raise NotImplementedError("ONNX conversion is not yet implemented")

    elif output_format == "quantized":
        print("Quantizing model...")
        # TODO: Implement quantization
        # This would use bitsandbytes or torch.quantization
        print("Quantization not yet implemented")
        raise NotImplementedError("Model quantization is not yet implemented")

    else:
        raise ValueError(f"Unknown output format: {output_format}")

    print(f"Model packaged successfully at: {output_dir}")

    return output_dir


@ray.remote(num_cpus=2, memory=8 * 1024 * 1024 * 1024)  # 8Gi
def create_model_metadata(
    model_path: str,
    training_config: Dict,
    training_metrics: Dict,
) -> str:
    """
    Create metadata file for trained model.

    Args:
        model_path: Path to trained model
        training_config: Training configuration used
        training_metrics: Training metrics and results

    Returns:
        Path to metadata file
    """
    import json
    import os
    from datetime import datetime

    metadata = {
        "model_path": model_path,
        "created_at": datetime.now().isoformat(),
        "training_config": training_config,
        "training_metrics": training_metrics,
        "framework": "transformers + peft + accelerate",
        "format": "huggingface",
    }

    metadata_path = os.path.join(model_path, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Created metadata file: {metadata_path}")

    return metadata_path


@ray.remote(num_cpus=4, num_gpus=1, memory=16 * 1024 * 1024 * 1024)  # 16Gi, 1 GPU
def merge_lora_weights(
    base_model_path: str,
    lora_adapter_path: str,
    output_dir: str,
) -> str:
    """
    Merge LoRA adapter weights with base model.

    This is useful for deployment when you want a single model
    instead of base model + adapter.

    Args:
        base_model_path: Path to base model
        lora_adapter_path: Path to LoRA adapter weights
        output_dir: Output directory for merged model

    Returns:
        Path to merged model
    """
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model from: {base_model_path}")
    print(f"Loading LoRA adapter from: {lora_adapter_path}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="auto"
    )

    # Load and merge LoRA weights
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model = model.merge_and_unload()

    # Save merged model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_dir)

    print(f"Merged model saved to: {output_dir}")

    return output_dir
