import evaluate
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
dataset = load_from_disk("data")


def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=150)


dataset = dataset.map(tokenize, batched=True).remove_columns(["sentence"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset.set_format(
    "torch", device=device, columns=["input_ids", "attention_mask", "labels"]
)

### input_ids must be the first column
dataset = dataset.map(lambda batch: {"new_labels": batch["labels"]}, batched=True)
dataset = dataset.remove_columns("labels")
dataset = dataset.rename_column("new_labels", "labels")


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

peft_config = LoraConfig(
    TaskType.SEQ_CLS, "vinai/phobert-base-v2", r=8, lora_alpha=8, lora_dropout=0.1
)
model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base-v2", num_labels=2
)
model = get_peft_model(model, peft_config)

args = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=64,
    per_device_train_batch_size=64,
    optim="adamw_torch_fused",
    tf32=True,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=10,
    logging_strategy="epoch",
    save_strategy="epoch",
    dataloader_num_workers=10,
    remove_unused_columns=False,
)

args.set_dataloader(auto_find_batch_size=True)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()

merged_model = model.merge_and_unload()
merged_model.save_pretrained("SentimentPhoBERT-LoRA-150/")
