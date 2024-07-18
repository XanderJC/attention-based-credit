# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional
from IPython import embed
import tyro
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from trl import RewardConfig, RewardTrainer, is_xpu_available
import numpy as np

tqdm.pandas()


def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


@dataclass
class ScriptArguments:
    # model_name: str = "./gpt2-sft-tldr"
    # model_name: str = "EleutherAI/gpt-neo-1.3B"
    model_name: str = "microsoft/phi-2"
    """the model name"""
    dataset_name: str = "openai/summarize_from_feedback"
    """the dataset name"""
    dataset_text_field: str = "text"
    """the text field of the dataset"""
    eval_split: str = "validation[:10%]"
    """the dataset split to evaluate on; default to 'none' (no evaluation)"""
    load_in_8bit: bool = True
    """load the model in 8 bits precision"""
    load_in_4bit: bool = False
    """load the model in 4 bits precision"""
    trust_remote_code: bool = True
    """Enable `trust_remote_code`"""
    reward_config: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            output_dir="phi2-sft-tldr",
            per_device_train_batch_size=4,
            num_train_epochs=1,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=1.41e-5,
            report_to="wandb",
            remove_unused_columns=False,
            optim="adamw_torch",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=1000,
            max_length=512,
            save_strategy="no",
            # optimize_device_cache=True,
        )
    )
    use_peft: bool = True
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=32,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["scores"],
        ),
    )


args = ScriptArguments()
args.reward_config.evaluation_strategy = "epoch" if args.eval_split != "none" else "no"


# Step 1: Load the model
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif args.load_in_8bit or args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit
    )
    # Copy the model to each device
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
else:
    device_map = None
    quantization_config = None

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=args.trust_remote_code,
    num_labels=1,
)
model.config.pad_token_id = model.config.eos_token_id
# Step 2: Load the dataset and pre-process it
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset = load_dataset(args.dataset_name, "comparisons", split="train")


def preprocess_function(example):
    choice, post, sum1, sum2 = (
        example["choice"],
        example["info"]["post"],
        example["summaries"][0]["text"],
        example["summaries"][1]["text"],
    )
    text1 = f"### Text to Summarize: {post}\n ### Summary: {sum1}"
    text2 = f"### Text to Summarize: {post}\n ### Summary: {sum2}"
    if choice == 1:
        text1, text2 = text2, text1
    tokenized_chosen = tokenizer(text1)
    tokenized_rejected = tokenizer(text2)

    example["input_ids_chosen"] = tokenized_chosen["input_ids"]
    example["attention_mask_chosen"] = tokenized_chosen["attention_mask"]
    example["input_ids_rejected"] = tokenized_rejected["input_ids"]
    example["attention_mask_rejected"] = tokenized_rejected["attention_mask"]
    return example


# Preprocess the dataset and filter out examples that are longer than args.max_length
train_dataset = train_dataset.map(
    preprocess_function,
    batched=False,
    num_proc=4,
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= args.reward_config.max_length
    and len(x["input_ids_rejected"]) <= args.reward_config.max_length
)
# train_dataset = train_dataset.remove_columns(["info", "summaries", "choice", "worker", "batch", "split", "extra"])

if args.eval_split == "none":
    eval_dataset = None
else:
    eval_dataset = load_dataset(args.dataset_name, "comparisons", split=args.eval_split)

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=False,
        num_proc=4,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= args.reward_config.max_length
        and len(x["input_ids_rejected"]) <= args.reward_config.max_length
    )


# Step 4: Define the LoraConfig
if args.use_peft:
    peft_config = args.peft_config
else:
    peft_config = None

# Step 5: Define the Trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args.reward_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()
trainer.push_to_hub()
