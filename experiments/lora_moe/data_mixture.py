"""
UltraChat-200k dataset loading and tokenization.

Dataset: HuggingFaceH4/ultrachat_200k
Format:  multi-turn conversations in 'messages' field
         [{"role": "user"|"assistant", "content": str}, ...]

Strategy: flatten each conversation to a single prompt+response by
treating the first user turn as the instruction and the first assistant
turn as the response (subsequent turns are discarded for simplicity).
Label masking follows the same convention as data.py: only the response
portion contributes to the cross-entropy loss.
"""

from __future__ import annotations

from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Tokenization (mirrors _tokenize_example from data.py)
# ---------------------------------------------------------------------------

def _flatten_conversation(messages: list) -> tuple[str, str]:
    """
    Extract (instruction, response) from a multi-turn message list.

    Takes the first user message as instruction and the first assistant
    reply as the response.  If the conversation doesn't have at least one
    of each, returns empty strings.
    """
    instruction = ""
    response = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and not instruction:
            instruction = content
        elif role == "assistant" and not response:
            response = content
        if instruction and response:
            break
    return instruction, response


def _tokenize_ultrachat_example(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict:
    messages = example.get("messages", [])
    instruction, response = _flatten_conversation(messages)

    if not instruction or not response:
        # Return a minimal valid (but empty) example — will be filtered later
        return {"input_ids": [], "attention_mask": [], "labels": [], "_has_labels": False}

    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:\n"
    )
    full_text = prompt + response + tokenizer.eos_token

    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )["input_ids"]

    # Mask prompt tokens so only the response contributes to loss
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    labels = labels[: len(full_ids)]

    # Check that at least one token is unmasked (response not completely truncated)
    has_labels = any(l != -100 for l in labels)

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "_has_labels": has_labels,
    }


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_ultrachat(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 1024,
    val_split: float = 0.05,
    max_samples: Optional[int] = None,
    seed: int = 42,
    num_proc: int = 4,
) -> DatasetDict:
    """
    Download HuggingFaceH4/ultrachat_200k, tokenize, and return
    train/validation splits.

    The dataset ships with its own train_sft / test_sft splits; we
    combine them into a single pool and re-split according to val_split
    so that max_samples applies uniformly.

    Args:
        tokenizer    : any HuggingFace tokenizer
        max_length   : sequence truncation length
        val_split    : fraction of data to use for validation
        max_samples  : if set, subsample (useful for quick debug runs)
        seed         : random seed for splits
        num_proc     : parallel tokenisation workers

    Returns:
        DatasetDict with "train" and "validation" keys.
    """
    # UltraChat-200k has "train_sft" and "test_sft" splits
    train_raw: Dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="train_sft"
    )
    test_raw: Dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="test_sft"
    )

    # Concatenate and optionally subsample
    from datasets import concatenate_datasets
    raw = concatenate_datasets([train_raw, test_raw]).shuffle(seed=seed)

    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    splits: DatasetDict = raw.train_test_split(test_size=val_split, seed=seed)
    splits = DatasetDict(train=splits["train"], validation=splits["test"])

    tokenized = splits.map(
        lambda ex: _tokenize_ultrachat_example(ex, tokenizer, max_length),
        batched=False,
        remove_columns=splits["train"].column_names,
        num_proc=num_proc,
        desc="Tokenising UltraChat-200k",
    )

    # Filter out empty examples and all-masked examples
    # (conversations without user+assistant, or where the prompt fills max_length
    # leaving no room for any response token — those produce NaN CE loss)
    tokenized = tokenized.filter(
        lambda ex: ex["_has_labels"] and len(ex["input_ids"]) > 0,
        num_proc=num_proc,
        desc="Filtering invalid examples",
    )

    # Remove the helper column before returning
    tokenized = tokenized.remove_columns("_has_labels")

    tokenized.set_format("torch")
    return tokenized
