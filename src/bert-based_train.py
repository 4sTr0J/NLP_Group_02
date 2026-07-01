from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

project_root = Path(r"G:\My Drive\NLP project").resolve()
data_dir = project_root / "data"

train_path = data_dir / "train_data.csv"
test_path = data_dir / "test_data.csv"

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

# Train the model 3 times.
training_args = TrainingArguments(
    output_dir=str(project_root / "models" / "bert-based_model"),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
)

train_dataset = load_dataset("csv", data_files=str(train_path), split="train")
test_dataset = load_dataset("csv", data_files=str(test_path), split="train")


def preprocess_function(examples):
    tokenized = tokenizer(examples["code"], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = examples["target"]
    return tokenized


train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["code", "target"])
test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=["code", "target"])

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)

trainer.train()
trainer.save_model(str(project_root / "models" / "bert-based_model"))
