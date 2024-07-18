from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
import numpy as np

# from peft import prepare_model_for_kbit_training
# from peft import LoraConfig, get_peft_model

set_seed(41310)


def preprocess(examples):
    # randomly decide if posisive or negative
    bin = np.random.binomial(1, 0.5)
    if bin == 1:
        examples["label"] = 1
        examples["text"] = f"{examples['prompt']}\n\n{examples['chosen']}"
    else:
        examples["label"] = 0
        examples["text"] = f"{examples['prompt']}\n\n{examples['rejected']}"

    examples["input_ids"] = tokenizer(
        examples["text"],
        truncation=True,
        # padding="max_length",
        return_tensors="pt",
        # max_length=512,
    )["input_ids"][0]
    return examples


def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


model_name = "gpt2"

model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"
model.config.pad_token_id = model.config.eos_token_id

max_length = 512

ds = load_dataset("CarperAI/openai_summarize_comparisons", split="train[:50%]")
ds = ds.map(preprocess, batched=False)
ds.set_format(type="torch")
ds = ds.filter(lambda x: len(x["input_ids"]) < max_length, batched=False)

ds_test = load_dataset("CarperAI/openai_summarize_comparisons", split="test[:10%]")
ds_test = ds_test.map(preprocess, batched=False)
ds_test.set_format(type="torch")
ds_test = ds_test.filter(lambda x: len(x["input_ids"]) < max_length, batched=False)

print(ds[0])


training_args = TrainingArguments(
    num_train_epochs=10,
    output_dir="gpt2-rm-tldr",
    # push_to_hub=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy=None,
)

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    args=training_args,
    train_dataset=ds,
    eval_dataset=ds_test,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()
