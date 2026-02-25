"""
Example: Chat Model Finetuning with ShareGPT Format

This example demonstrates how to finetune a chat model using the ShareGPT
conversation format.

Data format:
{
    "conversations": [
        {"from": "human", "value": "What is the capital of France?"},
        {"from": "gpt", "value": "The capital of France is Paris."},
        {"from": "human", "value": "What about Germany?"},
        {"from": "gpt", "value": "The capital of Germany is Berlin."}
    ]
}
"""

import os
from llama_finetune.configs.hyperparameters import HYPERPARAMETERS_CHAT, create_custom_config
from llama_finetune.workflows.finetune_pipeline import run_finetuning_workflow


def main():
    # ==================== Configure Training ====================
    # Use environment variables for storage configuration
    # export STORAGE_BACKEND=s3
    # export STORAGE_BASE_PATH=s3://your-bucket/models
    # export AWS_ACCESS_KEY_ID=...
    # export AWS_SECRET_ACCESS_KEY=...

    training_config = create_custom_config(
        # Model - Using chat-tuned base model
        MODEL_ID="llama2-7b-chat-custom-v1",
        BASE_MODEL="meta-llama/Llama-2-7b-chat-hf",
        MODEL_DESCRIPTION="Llama-2 7B Chat model finetuned on custom conversations",

        # Data - ShareGPT format
        INPUT_DATA_PATH="s3://your-bucket/data/conversations.json",
        DATASET_FORMAT="sharegpt",
        MAX_SEQ_LENGTH=1024,  # Longer sequences for multi-turn conversations
        VALIDATION_SPLIT=0.05,  # Smaller validation split

        # Finetuning - LoRA for efficiency
        FINETUNING_TYPE="lora",
        LORA_RANK=16,  # Higher rank for better quality
        LORA_ALPHA=32,
        LORA_DROPOUT=0.05,
        LORA_TARGET_MODULES=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

        # Training - Fewer epochs for chat models
        BATCH_SIZE=2,
        GRADIENT_ACCUMULATION_STEPS=8,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=1,  # Chat models often need less training
        WARMUP_RATIO=0.05,

        # Hardware
        NUM_GPUS=8,
        USE_DEEPSPEED=True,
        DEEPSPEED_STAGE=2,
        USE_BF16=True,

        # Checkpointing
        SAVE_STEPS=200,
        EVAL_STEPS=200,
    )

    # ==================== Run Training ====================
    print("=" * 80)
    print("Chat Model Finetuning Example - ShareGPT Format")
    print("=" * 80)
    print(f"Model: {training_config['BASE_MODEL']}")
    print(f"Data: {training_config['INPUT_DATA_PATH']}")
    print(f"Format: {training_config['DATASET_FORMAT']}")
    print(f"Max Seq Length: {training_config['MAX_SEQ_LENGTH']}")
    print("=" * 80)

    results = run_finetuning_workflow(training_config)

    # ==================== Results ====================
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Model Path: {results['model_path']}")
    print(f"Adapter Path: {results.get('adapter_path', 'N/A')}")
    print("=" * 80)

    # ==================== Test the Model ====================
    print("\nTo test the chat model:")
    print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "{training_config['BASE_MODEL']}",
    device_map="auto",
    torch_dtype="auto"
)
model = PeftModel.from_pretrained(base_model, "{results.get('adapter_path', results['model_path'])}")
tokenizer = AutoTokenizer.from_pretrained("{results['model_path']}")

# Chat
def chat(message, history=""):
    prompt = f"{{history}}<s>[INST] {{message}} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response

# Example
response = chat("What is machine learning?")
print(response)
    """)


if __name__ == "__main__":
    main()
