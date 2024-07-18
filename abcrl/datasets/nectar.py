import re
from datasets import load_dataset
from transformers import LlamaTokenizer, AutoTokenizer


def build_nectar_dataset(
    config,
    dataset_name="berkeley-nest/Nectar",
):
    """
    Build dataset for training.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    prompt = "Below is an instruction from a Human that describes a task. Write a response as the Assistant that appropriately completes the request."
    try:
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name, use_fast=False)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)

    ds = load_dataset(dataset_name, split="train")
    ds = ds.filter(lambda x: x["turns"] == 1, batched=False)
    ds = ds.filter(lambda x: "NAME" not in x["prompt"], batched=False)

    def tokenize(sample):
        pattern = r"\n\n(?=Human:|Assistant:)"
        # Replacement string
        replacement = " ###"
        # Using regex to replace the matched patterns
        modified = re.sub(pattern, replacement, sample["prompt"])
        modified = prompt + modified
        sample["input_ids"] = tokenizer.encode(modified)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds = ds.filter(lambda x: len(x["input_ids"]) < 256, batched=False)
    ds.set_format(type="torch")
    return ds
