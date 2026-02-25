"""
Alpaca dataset loading and tokenization.

Dataset: tatsu-lab/alpaca  (52K instruction-following examples)
Format:  {"instruction": str, "input": str, "output": str}

Prompt template (standard Alpaca format):
    Below is an instruction that describes a task[, paired with an input
    that provides further context]. Write a response that appropriately
    completes the request.

    ### Instruction:
    {instruction}

    [### Input:
    {input}]

    ### Response:
    {output}

Labels: full sequence with the prompt portion masked to -100
        (only the Response part contributes to loss).
"""

from __future__ import annotations

from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

ALPACA_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def _build_prompt(instruction: str, inp: str) -> str:
    if inp.strip():
        return ALPACA_PROMPT_WITH_INPUT.format(instruction=instruction, input=inp)
    return ALPACA_PROMPT_NO_INPUT.format(instruction=instruction)


def _tokenize_example(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict:
    prompt = _build_prompt(example["instruction"], example.get("input", ""))
    full_text = prompt + example["output"] + tokenizer.eos_token

    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )["input_ids"]

    # Mask prompt tokens in labels so loss only covers the response
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    # Truncate labels to match input_ids length (in case prompt alone fills max_length)
    labels = labels[: len(full_ids)]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def load_alpaca(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    val_split: float = 0.05,
    max_samples: Optional[int] = None,
    seed: int = 42,
    num_proc: int = 4,
) -> DatasetDict:
    """
    Download tatsu-lab/alpaca, tokenize, and return train/validation splits.

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
    raw: Dataset = load_dataset("tatsu-lab/alpaca", split="train")

    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    splits: DatasetDict = raw.train_test_split(test_size=val_split, seed=seed)
    splits = DatasetDict(train=splits["train"], validation=splits["test"])

    tokenized = splits.map(
        lambda ex: _tokenize_example(ex, tokenizer, max_length),
        batched=False,
        remove_columns=splits["train"].column_names,
        num_proc=num_proc,
        desc="Tokenising Alpaca",
    )

    tokenized.set_format("torch")
    return tokenized
