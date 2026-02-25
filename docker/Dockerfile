# Dockerfile for PyTorch LLM Training (transformers + peft + accelerate + Ray)
# LLM Finetune Framework - External Version

# Use NVIDIA CUDA base image with cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 -s /bin/bash mluser
USER mluser
WORKDIR /home/mluser

# Set up CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Configure NCCL for multi-GPU training
ENV NCCL_DEBUG=INFO
ENV NCCL_SOCKET_IFNAME=eth0
ENV NCCL_IB_DISABLE=1

# Set HuggingFace cache directory
ENV HF_HOME=/home/mluser/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/mluser/.cache/huggingface
ENV HF_DATASETS_CACHE=/home/mluser/.cache/huggingface/datasets

# Create directories for cache and data
RUN mkdir -p \
    /home/mluser/.cache/huggingface \
    /home/mluser/data \
    /home/mluser/output \
    /home/mluser/workspace

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install core ML libraries
RUN pip install --no-cache-dir \
    transformers>=4.38.1 \
    peft>=0.9.0 \
    accelerate>=0.27.2 \
    datasets>=2.17.0 \
    bitsandbytes>=0.42.0

# Install distributed training libraries
RUN pip install --no-cache-dir \
    deepspeed>=0.13.0 \
    ray[default]>=2.9.0

# Install storage and utilities
RUN pip install --no-cache-dir \
    fsspec>=2023.1.0 \
    s3fs>=2023.1.0 \
    tqdm>=4.65.0 \
    pyyaml>=6.0

# Optional: Install logging tools (uncomment if needed)
# RUN pip install --no-cache-dir \
#     wandb>=0.15.0 \
#     tensorboard>=2.13.0 \
#     mlflow>=2.7.0

# Copy the application code
COPY --chown=mluser:mluser . /home/mluser/workspace/llama-finetune

# Install the package in editable mode
WORKDIR /home/mluser/workspace/llama-finetune
RUN pip install --no-cache-dir -e .

# Set working directory
WORKDIR /home/mluser/workspace

# Expose ports for Ray dashboard and TensorBoard
EXPOSE 8265 6006

# Default command
ENTRYPOINT ["/bin/bash"]
