"""
Train model using the final training set and save locally.

This script:
1. Loads data from final_training_set/
2. Trains the model
3. Saves model locally so you can use it multiple times
"""

import json
import sys
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np

print("="*80)
print("TRAIN MODEL WITH FINAL DATASET")
print("="*80)

# Paths
final_dir = Path(r"F:\DATA STORAGE\AGG_PACKET\final_training_set")
model_output_dir = Path(__file__).parent.parent.parent / "saved_models" / "depression_classifier_final"

# Check data exists
train_file = final_dir / "train.json"
val_file = final_dir / "val.json"
test_file = final_dir / "test.json"

if not train_file.exists():
    raise FileNotFoundError(f"Training file not found: {train_file}")

print(f"\nLoading training data from: {final_dir}")
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
print(f"\nLabel distribution (train):")
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
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # Binary classification
)

# Update label mapping in model config
model.config.id2label = {0: "non-depression", 1: "depression"}
model.config.label2id = {"non-depression": 0, "depression": 1}

print(f"  Model: {model_name}")
print(f"  Labels: {model.config.id2label}")

# Tokenize datasets
print("\nTokenizing datasets...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
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

# Training arguments
print("\n" + "="*80)
print("TRAINING CONFIGURATION")
print("="*80)
print("""
Training parameters:
  - Learning rate: 2e-5
  - Batch size: 4 per device
  - Epochs: 3
  - Evaluation: After each epoch
  - Save strategy: Best model (based on F1 score)
  - Output directory: saved_models/depression_classifier_final/
""")

training_args = TrainingArguments(
    output_dir=str(model_output_dir),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir=str(model_output_dir / "logs"),
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=3,  # Keep only best 3 checkpoints
    report_to="none",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Train
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"\nThis will save the model to: {model_output_dir}")
print("Training may take several hours on CPU...")
print("\nPress Ctrl+C to interrupt (model will be saved at checkpoints)")

response = input("\nStart training? (y/n): ").strip().lower()
if response != 'y':
    print("Cancelled.")
    sys.exit(0)

try:
    trainer.train()
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"\nTest set results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save final model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "training_examples": len(train_data),
        "validation_examples": len(val_data),
        "test_examples": len(test_data),
        "test_results": test_results,
        "label_mapping": dict(model.config.id2label),
        "training_args": {
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "epochs": training_args.num_train_epochs,
        }
    }
    
    info_file = model_output_dir / "training_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Model saved to: {model_output_dir}")
    print("\nFiles saved:")
    print("  - Model weights")
    print("  - Tokenizer")
    print("  - Config")
    print("  - Training info")
    print("\nYou can now load and use this model multiple times!")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted.")
    print(f"Checkpoints may be available in: {model_output_dir}")
    print("You can resume training or use the latest checkpoint.")
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nModel location: {model_output_dir}")
print("\nTo load and use the model later:")
print(f"  from transformers import AutoModelForSequenceClassification, AutoTokenizer")
print(f"  model = AutoModelForSequenceClassification.from_pretrained(r'{model_output_dir}')")
print(f"  tokenizer = AutoTokenizer.from_pretrained(r'{model_output_dir}')")




