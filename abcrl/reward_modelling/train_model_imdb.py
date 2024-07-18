from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np

# from peft import prepare_model_for_kbit_training
# from peft import LoraConfig, get_peft_model


def tokenize(examples):
    outputs = tokenizer(examples["text"], truncation=True, padding=True)
    return outputs


def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


model_name = "lvwerra/gpt2-imdb"

model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"
model.config.pad_token_id = model.config.eos_token_id

ds = load_dataset("imdb")
tokenized_ds = ds.map(tokenize, batched=True)

print(tokenized_ds["train"][0])

training_args = TrainingArguments(
    num_train_epochs=2,
    output_dir="gpt2-rm-imdb",
    # push_to_hub=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
# trainer.push_to_hub()
