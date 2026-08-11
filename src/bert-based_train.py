from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

project_root = Path(r"D:\NLP project").resolve()
data_dir     = project_root / "data"
model_dir    = project_root / "models" / "codebert"   # FIX: was "bert-based_model", now matches predict.py

train_path = data_dir / "train_data.csv"
test_path  = data_dir / "test_data.csv"

# Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
config    = AutoConfig.from_pretrained("microsoft/codebert-base")
config.num_labels = 2
model     = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base", config=config
)

import pandas as pd
from datasets import Dataset

# Load full datasets using pandas for robustness and convert to HF Dataset
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
test_dataset  = Dataset.from_pandas(test_df, preserve_index=False)

def preprocess_function(examples):
    tokenized = tokenizer(
        examples["code"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized["labels"] = examples["target"]
    return tokenized

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["code", "target"], load_from_cache_file=False)
test_dataset  = test_dataset.map(preprocess_function,  batched=True, remove_columns=["code", "target"], load_from_cache_file=False)

# Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    probs = np.exp(logits[:, 1]) / np.sum(np.exp(logits), axis=1)  # softmax for class 1
    return {
        "accuracy":  accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall":    recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1":        f1_score(labels, predictions, average="weighted", zero_division=0),
        "roc_auc":   roc_auc_score(labels, probs),
    }

training_args = TrainingArguments(
    output_dir=str(model_dir),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",        # FIX: was missing — now evaluates after every epoch
    save_strategy="epoch",        # FIX: was save_steps=10_000 which may never trigger on small datasets
    save_total_limit=2,
    load_best_model_at_end=True,  # saves the best checkpoint automatically
    metric_for_best_model="f1",
    logging_steps=50,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),                    # faster training if GPU is available; safe to keep on CPU too
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save final model and tokenizer to codebert folder
trainer.save_model(str(model_dir))
tokenizer.save_pretrained(str(model_dir))   # FIX: was missing — tokenizer must be saved alongside model

print(f"\nModel and tokenizer saved to: {model_dir}")