# Migration Guide: LinkedIn Version → External Version

This guide helps you migrate from the LinkedIn internal `openconnect-llamafactory` to the external `llama-finetune` framework.

## Key Differences

### 1. Orchestration: Flyte + OpenConnect → Ray

**Before (LinkedIn):**
```python
from flytekit import workflow, task
import openconnect.pcv2 as oc
from openconnect.core.openconnect_context import OpenConnectContext

@workflow
def LLMFinetune_workflow():
    resource_upload_outputs = oc.resource_upload(...)
    # ... rest of workflow
```

**After (External):**
```python
import ray
from llama_finetune.workflows.finetune_pipeline import run_finetuning_workflow

# Simply call the workflow
results = run_finetuning_workflow(training_config)
```

### 2. Configuration: Per-User Settings → Environment Variables

**Before (LinkedIn):**
```python
from openconnect.predefined_user_settings import get_user_settings

user_name = os.environ.get("USER", "default_user")
user_settings = get_user_settings(user_name)
openconnect_context = OpenConnectContext.create(
    proxy_as=user_settings.userToProxy,
    base_pipeline_dir=user_settings.homeDir,
    ...
)
```

**After (External):**
```bash
# Set environment variables
export STORAGE_BACKEND=s3
export STORAGE_BASE_PATH=s3://your-bucket/models
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

```python
from llama_finetune.configs.config import get_config

# Configuration is loaded from environment
config = get_config()
```

### 3. Storage: HDFS → S3/Azure/GCS/Local

**Before (LinkedIn):**
```python
input_data_path = "/user/username/llm-experiments/data.json"  # HDFS
output_base_dir = f"{user_settings.homeDir}/{model_id}"
```

**After (External):**
```python
# S3
input_data_path = "s3://your-bucket/data/training_data.json"

# Or local
input_data_path = "/path/to/local/data.json"

# Output paths are managed by config
config = get_config()
output_dir = config.get_output_dir(model_id)
```

### 4. Build System: Gradle + distgradle → setuptools + pip

**Before (LinkedIn):**
```bash
mint build
mint oc-env
./gradlew check
```

**After (External):**
```bash
pip install -e .
pytest
black src/
```

### 5. Container Images: LinkedIn Registry → Public NVIDIA

**Before (LinkedIn):**
```dockerfile
FROM container-image-registry.corp.linkedin.com:8083/lps-image/linkedin/dl-base-images/cm2-py310-cuda11
```

**After (External):**
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```

### 6. Workflow Registration: openconnect.json → Direct Python

**Before (LinkedIn):**
```json
{
  "LLMFinetune_workflow": {
    "implementation": "openconnect.flows.LLMFinetunePipeline:LLMFinetune_workflow",
    ...
  }
}
```

**After (External):**
```python
# No registration needed, just import and call
from llama_finetune import run_finetuning_workflow

results = run_finetuning_workflow(config)
```

## Migration Steps

### Step 1: Update Dependencies

**Before (LinkedIn `pinned.txt`):**
```
openconnect-lib-pcv2==1.8.542
openconnect-lib-core==1.1.279
flytekit>=1.10.0
transformers>=4.38.1
...
```

**After (External `pyproject.toml`):**
```toml
[project]
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.38.1",
    "peft>=0.9.0",
    "accelerate>=0.27.2",
    "ray[default]>=2.9.0",
    "s3fs>=2023.1.0",
    ...
]
```

Install:
```bash
pip install -e .
```

### Step 2: Update Configuration

**Before:**
```python
from openconnect.predefined_user_settings import PerUserSettings

USER_SETTINGS = {
    "user1": PerUserSettings(
        homeDir="/user/user1/llm-experiments",
        userToProxy="grwthrel",
        email="user1@linkedin.com",
    )
}
```

**After:**
Set environment variables:
```bash
export STORAGE_BACKEND=s3
export STORAGE_BASE_PATH=s3://your-bucket/llm-experiments
export AWS_REGION=us-east-1
export PROJECT_NAME=llm-finetune
```

Or configure in code:
```python
from llama_finetune.configs.config import Config, StorageConfig

config = Config(
    storage=StorageConfig(
        backend="s3",
        base_path="s3://your-bucket/llm-experiments",
    ),
    project_name="llm-finetune",
)
```

### Step 3: Update Workflow Code

**Before (LinkedIn Flyte):**
```python
@workflow
def LLMFinetune_workflow() -> None:
    resource_upload_outputs = oc.resource_upload(...)

    data_prep_output = prepare_training_data(
        input_data_path=input_data_path,
        output_dir=f"{output_base_dir}/data",
        ...
    )

    training_output = transformers_trainer(
        model_name_or_path=base_model,
        train_file=data_prep_output["train_file"],
        ...
    )

    final_model_path = package_model_output(...)
```

**After (External Ray):**
```python
from llama_finetune import run_finetuning_workflow
from llama_finetune.configs.hyperparameters import HYPERPARAMETERS_LORA

config = HYPERPARAMETERS_LORA.copy()
config.update({
    "INPUT_DATA_PATH": "s3://bucket/data.json",
    "BASE_MODEL": "meta-llama/Llama-2-7b-hf",
})

results = run_finetuning_workflow(config)
```

### Step 4: Update Task Decorators

**Before (LinkedIn Flyte tasks):**
```python
@task(
    container_image=PYTORCH_IMAGE,
    requests=Resources(cpu="30", mem="360Gi", gpu="8"),
    proxy_as=PROXY_AS,
)
def transformers_trainer(...):
    ...
```

**After (External Ray tasks):**
```python
@ray.remote(num_cpus=30, num_gpus=8, memory=360 * 1024 * 1024 * 1024)
def transformers_trainer(...):
    ...
```

### Step 5: Update Data Paths

**Before (HDFS):**
```python
INPUT_DATA_PATH = "/user/username/data/training_data.json"
```

**After (S3 or local):**
```python
# S3
INPUT_DATA_PATH = "s3://your-bucket/data/training_data.json"

# Or local
INPUT_DATA_PATH = "/home/user/data/training_data.json"
```

### Step 6: Update Build and Deployment

**Before (LinkedIn):**
```bash
# Build
mint build

# Run
mldev run LLMFinetune_workflow
```

**After (External):**
```bash
# Install
pip install -e .

# Run
python -m llama_finetune.workflows.finetune_pipeline --config lora

# Or in Python
python examples/instruction_tuning.py
```

## Code Mapping Reference

| LinkedIn Component | External Component | Status |
|-------------------|-------------------|--------|
| `openconnect.pcv2.resource_upload()` | Removed (not needed) | ❌ Removed |
| `OpenConnectContext` | `Config` | ✅ Replaced |
| `@workflow` (Flyte) | Python function | ✅ Replaced |
| `@task` (Flyte) | `@ray.remote` | ✅ Replaced |
| `predefined_user_settings.py` | `configs/config.py` | ✅ Replaced |
| `base_config.py` | `configs/hyperparameters.py` | ✅ Kept (updated) |
| `data_preparation.py` | `tasks/data_preparation.py` | ✅ Kept (updated) |
| `training.py` | `tasks/training.py` | ✅ Kept (updated) |
| `model_packaging.py` | `tasks/model_packaging.py` | ✅ Kept (updated) |
| HDFS paths | S3/local paths | ✅ Replaced |
| Gradle build | pip install | ✅ Replaced |
| LinkedIn containers | NVIDIA public images | ✅ Replaced |

## Feature Parity

✅ **Available in external version:**
- LoRA, QLoRA, and full finetuning
- Multi-GPU distributed training
- DeepSpeed ZeRO-2/3
- Alpaca and ShareGPT data formats
- S3, Azure Blob, GCS, and local storage
- Model packaging and metadata
- LoRA adapter merging
- Comprehensive examples and documentation

❌ **Not available (LinkedIn-specific):**
- OpenConnect resource upload
- LinkedIn HDFS integration
- LinkedIn service account proxying
- LinkedIn-specific monitoring and alerting

## Testing Your Migration

1. **Install the package:**
   ```bash
   cd llama-finetune-external
   pip install -e .
   ```

2. **Run tests:**
   ```bash
   pytest tests/
   ```

3. **Try a simple example:**
   ```bash
   # Set up credentials
   export STORAGE_BACKEND=local
   export STORAGE_BASE_PATH=./outputs

   # Run example
   python examples/local_training.py
   ```

4. **Verify output:**
   Check that models are saved to the expected location.

## Getting Help

- **Documentation:** See [README.md](README.md)
- **Examples:** Check `examples/` directory
- **Issues:** Report at repository issues page

## Migration Checklist

- [ ] Update dependencies (install new package)
- [ ] Set up environment variables (storage, AWS credentials, etc.)
- [ ] Update data paths (HDFS → S3/local)
- [ ] Remove OpenConnect-specific code
- [ ] Replace Flyte decorators with Ray
- [ ] Test locally with small dataset
- [ ] Run full training job
- [ ] Verify model outputs
- [ ] Update documentation and scripts
