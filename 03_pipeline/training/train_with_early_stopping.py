"""
Train model with early stopping for optimal results.

This script:
1. Trains for up to 10 epochs
2. Uses early stopping to prevent overfitting
3. Automatically saves the best model
4. Stops when validation performance plateaus
"""
import json
import sys
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np

print("="*80)
print("TRAIN MODEL WITH EARLY STOPPING")
print("="*80)

# Paths
final_dir = Path(r"F:\DATA STORAGE\AGG_PACKET\final_training_set")
model_output_dir = Path(__file__).parent.parent.parent / "saved_models" / "depression_classifier_early_stop"

# Create output directory
model_output_dir.mkdir(parents=True, exist_ok=True)

# Check data exists
train_file = final_dir / "train.json"
val_file = final_dir / "val.json"
test_file = final_dir / "test.json"

if not train_file.exists():
    raise FileNotFoundError(f"Training file not found: {train_file}")

print(f"\nTraining data: {final_dir}")
print(f"Model will be saved to: {model_output_dir}")

# Load datasets
print("\nLoading datasets...")
with open(train_file, 'r', encoding='utf-8') as f:
    train_data = json.load(f)
with open(val_file, 'r', encoding='utf-8') as f:
    val_data = json.load(f)
with open(test_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"  Train: {len(train_data):,} examples")
print(f"  Val: {len(val_data):,} examples")
print(f"  Test: {len(test_data):,} examples")

# Verify labels
from collections import Counter
train_labels = Counter(r['label'] for r in train_data)
print(f"\nLabel distribution:")
for label, count in sorted(train_labels.items()):
    label_name = {0: "non-depression", 1: "depression"}.get(label, f"unknown({label})")
    print(f"  Label {label} ({label_name}): {count:,}")

# Create HuggingFace datasets
print("\nCreating HuggingFace datasets...")
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

# Load model and tokenizer
print("\nLoading pretrained model...")
model_name = "rafalposwiata/deproberta-large-depression"
print(f"  Model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("[INFO] Loading model...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    print("[INFO] Model loaded successfully!")
except Exception as e:
    print(f"[WARNING] Error: {e}")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 2
    config.id2label = {0: "non-depression", 1: "depression"}
    config.label2id = {"non-depression": 0, "depression": 1}
    
    full_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_config(config)
    model.roberta.load_state_dict(full_model.roberta.state_dict())
    print("[INFO] Model created with new classification head!")

model.config.id2label = {0: "non-depression", 1: "depression"}
model.config.label2id = {"non-depression": 0, "depression": 1}

print(f"[INFO] Label mapping: {model.config.id2label}\n")

# Tokenize datasets
print("Tokenizing datasets...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=1000)
val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=1000)
test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=1000)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Training arguments with early stopping
print("\n" + "="*80)
print("TRAINING CONFIGURATION")
print("="*80)
print("""
Training parameters:
  - Learning rate: 2e-5
  - Batch size: 4 per device
  - Max epochs: 10 (will stop early if no improvement)
  - Early stopping: Stops if no improvement for 3 epochs
  - Evaluation: After each epoch
  - Save strategy: Best model (based on F1 score)
  - Output: saved_models/depression_classifier_early_stop/
""")

training_args = TrainingArguments(
    output_dir=str(model_output_dir),
    num_train_epochs=10,  # Maximum epochs (will stop early if needed)
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir=str(model_output_dir / "logs"),
    logging_steps=10,
    logging_first_step=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,  # Use best model, not last
    metric_for_best_model="f1",  # Monitor F1 score
    greater_is_better=True,
    save_total_limit=3,
    report_to="none",
    fp16=False,
    dataloader_pin_memory=False,
)

# Create trainer with early stopping callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if no improvement for 3 epochs
)

# Train
print("\n" + "="*80)
print("STARTING TRAINING WITH EARLY STOPPING")
print("="*80)
print("\nTraining will:")
print("  - Train for up to 10 epochs")
print("  - Stop early if validation F1 doesn't improve for 3 epochs")
print("  - Automatically save the best model")
print("\nPress Ctrl+C to stop manually (best model will still be saved)")
print("="*80 + "\n")

try:
    trainer.train()
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"\nTest set results:")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save final model
    print("\n" + "="*80)
    print("SAVING BEST MODEL")
    print("="*80)
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "training_examples": len(train_data),
        "validation_examples": len(val_data),
        "test_examples": len(test_data),
        "test_results": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in test_results.items()},
        "label_mapping": dict(model.config.id2label),
        "training_args": {
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "max_epochs": training_args.num_train_epochs,
            "early_stopping_patience": 3,
        }
    }
    
    info_file = model_output_dir / "training_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Best model saved to: {model_output_dir}")
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nThe model stopped automatically when validation performance plateaued.")
    print("The best model (highest F1 score) has been saved.")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")
    print(f"Best model saved in: {model_output_dir}")
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)




