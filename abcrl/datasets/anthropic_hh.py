import re
from datasets import load_dataset
from transformers import LlamaTokenizer, AutoTokenizer
import torch.utils.data as data
from trl import PPOConfig


def build_anthropic_dataset(
    config: PPOConfig,
    max_length: int = 256,
) -> data.Dataset:
    """
    Build dataset for training.

    Args:
        config: The configuration object for a PPOTrainer.
        max_length (int, optional): The maximum length of the input sequences. Defaults to 256.

    Returns:
        torch.utils.data.Dataset: The dataset for training.

    Raises:
        Exception: If the tokenizer fails to load.

    Example:
        >>> config = config = PPOConfig(model_name="VMware/open-llama-7b-open-instruct")
        >>> dataset = build_anthropic_dataset(config, max_length=512)
    """
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
    ds = ds.filter(lambda x: x["chosen"].count("Human:") == 1, batched=False)
    try:
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name, use_fast=False)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)

    def tokenize(sample):
        pattern = r"\s*Human:\s*(.*?)\s*Assistant:\s*"
        match = re.search(pattern, sample["chosen"], re.DOTALL)
        # prompt = f"Below is an instruction from a Human that describes a task. Write a response as the Assistant that appropriately completes the request. ###Human: {match.group(1).strip()} ###Assistant: "
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{match.group(1).strip()}\n\n### Response:"
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["rm_input_ids"] = tokenizer.encode(
            f"###Human: {match.group(1).strip()} ###Assistant: "
        )
        sample["rm_query"] = tokenizer.decode(sample["rm_input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds = ds.filter(lambda x: len(x["input_ids"]) < max_length, batched=False)
    ds.set_format(type="torch")
    return ds
