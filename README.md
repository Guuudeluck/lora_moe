# LLM Finetune

A production-ready framework for finetuning large language models using **transformers + peft + Ray**. Supports LoRA, QLoRA, and full finetuning with distributed training capabilities.

## Features

- 🚀 **Modern Stack**: transformers + peft + accelerate + Ray
- 🎯 **Multiple Finetuning Methods**: LoRA, QLoRA, and full finetuning
- ☁️ **Cloud Storage**: Native S3, Azure Blob, and GCS support
- 🔧 **Flexible Configuration**: Environment-based config with preset templates
- 📊 **Distributed Training**: DeepSpeed ZeRO-2/3 and multi-GPU support
- 🐳 **Containerized**: Docker images for reproducible training
- 🎨 **Multiple Data Formats**: Alpaca, ShareGPT, and custom formats

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llama-finetune.git
cd llama-finetune

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For all features (including Azure, GCS, logging)
pip install -e ".[all]"
```

### Basic Usage

```python
from llama_finetune.configs.hyperparameters import HYPERPARAMETERS_LORA
from llama_finetune.workflows.finetune_pipeline import run_finetuning_workflow

# Update configuration
config = HYPERPARAMETERS_LORA.copy()
config.update({
    "BASE_MODEL": "meta-llama/Llama-2-7b-hf",
    "INPUT_DATA_PATH": "s3://your-bucket/training-data.json",
    "MODEL_ID": "my-finetuned-model",
})

# Run finetuning
results = run_finetuning_workflow(config)
print(f"Model saved to: {results['model_path']}")
```

### CLI Usage

```bash
# Run finetuning with LoRA preset
python -m llama_finetune.workflows.finetune_pipeline --config lora

# Run evaluation
python -m llama_finetune.workflows.finetune_pipeline \
    --workflow evaluate \
    --model-path s3://bucket/models/llama-lora-v1 \
    --test-data s3://bucket/test-data.json

# Merge LoRA adapter with base model
python -m llama_finetune.workflows.finetune_pipeline \
    --workflow merge \
    --base-model meta-llama/Llama-2-7b-hf \
    --adapter-path s3://bucket/models/llama-lora-v1/adapter
```

## Configuration

### Environment Variables

Set these environment variables to configure the framework:

```bash
# Storage Configuration
export STORAGE_BACKEND=s3                          # s3, local, azure, or gcs
export STORAGE_BASE_PATH=s3://your-bucket/models   # Base path for outputs
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1

# Ray Configuration
export RAY_ADDRESS=auto                            # Ray cluster address (or None for local)
export NUM_GPUS=8                                  # Number of GPUs to use

# Training Configuration
export PROJECT_NAME=llm-finetune
export EXPERIMENT_NAME=experiment-1

# Optional: Logging
export ENABLE_WANDB=true
export WANDB_PROJECT=my-project
export ENABLE_TENSORBOARD=true
```

### Configuration Presets

The framework provides several preset configurations:

| Preset | Finetuning Type | GPUs | Batch Size | Use Case |
|--------|----------------|------|------------|----------|
| `base` | LoRA | 8 | 4 | General purpose |
| `lora` | LoRA | 4 | 4 | Memory efficient |
| `qlora` | QLoRA | 2 | 2 | Most memory efficient |
| `full` | Full | 8 | 2 | Best quality, most resources |
| `large` | LoRA | 8 | 2 | Large models (13B+) |
| `instruction` | LoRA | 8 | 4 | Instruction tuning |
| `chat` | LoRA | 8 | 4 | Chat model finetuning |

Use presets in your code:

```python
from llama_finetune.configs.hyperparameters import (
    HYPERPARAMETERS_LORA,
    HYPERPARAMETERS_QLORA,
    HYPERPARAMETERS_FULL,
    get_config_by_name
)

# Use preset directly
config = HYPERPARAMETERS_LORA

# Or get by name
config = get_config_by_name("qlora")

# Customize
config = create_custom_config(
    MODEL_ID="my-model",
    BASE_MODEL="mistralai/Mistral-7B-v0.1",
    LEARNING_RATE=1e-4,
)
```

## Data Formats

### Alpaca Format

```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
  }
]
```

### ShareGPT Format

```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is the capital of France?"},
      {"from": "gpt", "value": "The capital of France is Paris."}
    ]
  }
]
```

## Docker Usage

### Build Images

```bash
# GPU image with CUDA support
docker build -f docker/Dockerfile -t llama-finetune:gpu .

# Minimal CPU image
docker build -f docker/Dockerfile.minimal -t llama-finetune:cpu .
```

### Run Container

```bash
# Interactive mode
docker run --gpus all -it \
    -v $(pwd)/data:/home/mluser/data \
    -v $(pwd)/outputs:/home/mluser/output \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    llama-finetune:gpu

# Run training
docker run --gpus all \
    -v $(pwd)/data:/home/mluser/data \
    -v $(pwd)/outputs:/home/mluser/output \
    -e STORAGE_BACKEND=local \
    -e STORAGE_BASE_PATH=/home/mluser/output \
    llama-finetune:gpu \
    python -m llama_finetune.workflows.finetune_pipeline --config lora
```

## Advanced Features

### Distributed Training with DeepSpeed

Enable DeepSpeed in your configuration:

```python
config = {
    **HYPERPARAMETERS_FULL,
    "USE_DEEPSPEED": True,
    "DEEPSPEED_STAGE": 3,  # ZeRO-3 for maximum memory efficiency
    "GRADIENT_CHECKPOINTING": True,
}
```

### Custom Data Formats

Implement custom data processing in `tasks/data_preparation.py`:

```python
if dataset_format == "custom":
    # Add your custom processing logic
    processed_data = process_custom_format(data)
```

### Ray Cluster

Run on a Ray cluster:

```python
from llama_finetune.configs.config import Config, RayConfig

config = Config(
    ray=RayConfig(address="ray://cluster-address:10001"),
)

results = run_finetuning_workflow(training_config, config)
```

## Project Structure

```
llama-finetune/
├── src/llama_finetune/
│   ├── configs/
│   │   ├── config.py           # Environment-based configuration
│   │   └── hyperparameters.py  # Training presets
│   ├── tasks/
│   │   ├── data_preparation.py # Data loading and formatting
│   │   ├── training.py         # Training with transformers
│   │   └── model_packaging.py  # Model export and merging
│   └── workflows/
│       └── finetune_pipeline.py # Main orchestration
├── examples/                   # Example scripts
├── tests/                      # Test suite
├── docker/                     # Docker configurations
└── README.md
```

## Examples

See the `examples/` directory for complete examples:

- `examples/instruction_tuning.py` - Alpaca-format instruction tuning
- `examples/chat_finetuning.py` - ShareGPT chat model finetuning
- `examples/domain_adaptation.py` - Domain-specific finetuning

## Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 8GB+ RAM (for data preparation)
- 40GB+ GPU VRAM (for 7B model training with LoRA)
- 80GB+ GPU VRAM (for 7B full finetuning)

## Troubleshooting

### Out of Memory Errors

Try these solutions:
1. Use QLoRA instead of LoRA
2. Reduce batch size and increase gradient accumulation
3. Enable gradient checkpointing
4. Use DeepSpeed ZeRO-3

### S3 Connection Issues

Ensure credentials are set:
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

### Ray Connection Issues

For local mode, don't set `RAY_ADDRESS`:
```bash
unset RAY_ADDRESS
```

## Migration from LinkedIn Version

This is an external version of the LinkedIn internal `openconnect-llamafactory`. Key differences:

| Feature | LinkedIn Version | External Version |
|---------|-----------------|------------------|
| Orchestration | Flyte + OpenConnect | Ray |
| Build System | Gradle + distgradle | setuptools + pip |
| Storage | HDFS | S3, Azure, GCS, local |
| Container Registry | LinkedIn internal | Public (Docker Hub) |
| Base Images | dl-base-images | nvidia/cuda |

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Run `pytest` and ensure all tests pass
5. Submit a pull request

## License

Apache 2.0

## Support

- Issues: https://github.com/yourusername/llama-finetune/issues
- Documentation: https://github.com/yourusername/llama-finetune#readme

## Acknowledgments

Built on top of:
- [Transformers](https://github.com/huggingface/transformers) by Hugging Face
- [PEFT](https://github.com/huggingface/peft) by Hugging Face
- [Ray](https://github.com/ray-project/ray) by Anyscale
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) by Microsoft
