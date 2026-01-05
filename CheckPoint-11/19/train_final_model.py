"""
Train model using final_training_set and save locally for reuse.

This script:
1. Loads your final training set (train.json, val.json, test.json)
2. Trains the model
3. Saves model locally so you can use it multiple times without retraining
"""

import json
import sys
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np

print("="*80)
print("TRAIN MODEL - FINAL TRAINING SET")
print("="*80)

# Paths
final_dir = Path(r"F:\DATA STORAGE\AGG_PACKET\final_training_set")
model_output_dir = Path(__file__).parent.parent.parent / "saved_models" / "depression_classifier_final"

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
print("[INFO] Downloading tokenizer (if not cached)...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
print("[INFO] Tokenizer loaded!")

# Load model with 2 labels - ignore_mismatched_sizes will handle the classification head
print("[INFO] Downloading model (if not cached) - this may take a few minutes...")
print("[INFO] Model size: ~1.3GB - downloading now...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Binary: 0=non-depression, 1=depression
        ignore_mismatched_sizes=True  # Ignore the mismatch in classification head size
    )
    print("[INFO] Model loaded successfully!")
except Exception as e:
    print(f"[WARNING] Error loading with ignore_mismatched_sizes: {e}")
    print("[INFO] Trying alternative approach: loading base model and creating new head...")
    # Alternative: Load base model and manually create classification head
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 2
    config.id2label = {0: "non-depression", 1: "depression"}
    config.label2id = {"non-depression": 0, "depression": 1}
    
    # Load the full model first to get encoder weights
    full_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # Create new model with 2 labels
    model = AutoModelForSequenceClassification.from_config(config)
    # Copy encoder weights (roberta base)
    model.roberta.load_state_dict(full_model.roberta.state_dict())
    print("[INFO] Model created with new classification head!")

# Update label mapping
model.config.id2label = {0: "non-depression", 1: "depression"}
model.config.label2id = {"non-depression": 0, "depression": 1}

print(f"[INFO] Label mapping: {model.config.id2label}")
print("[INFO] Classification head will be trained from scratch\n")

# Tokenize datasets
print("\nTokenizing datasets (this may take a few minutes)...")
print("[INFO] This step processes all text data for training")
print("[INFO] Progress bars will show tokenization progress\n")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

print("[INFO] Tokenizing training set...")
train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=1000)
print("[INFO] Tokenizing validation set...")
val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=1000)
print("[INFO] Tokenizing test set...")
test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=1000)
print("[INFO] Tokenization complete!\n")

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
  - Checkpoints: Saved after each epoch
  - Output: saved_models/depression_classifier_final/
""")

training_args = TrainingArguments(
    output_dir=str(model_output_dir),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir=str(model_output_dir / "logs"),
    logging_steps=10,  # More frequent logging for live updates
    logging_first_step=True,  # Log the first step
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=3,  # Keep best 3 checkpoints
    report_to="none",
    fp16=False,  # Set to True if you have GPU with CUDA
    disable_tqdm=False,  # Show progress bars
    dataloader_pin_memory=False,  # Better for CPU
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
print("READY TO START TRAINING")
print("="*80)
print(f"\nModel will be saved to: {model_output_dir}")
print("\nTraining will:")
print("  - Train for 3 epochs")
print("  - Save checkpoints after each epoch")
print("  - Save the best model automatically")
print("  - Allow you to use the saved model multiple times")
print("\nNote: Training may take several hours on CPU")
print("\n" + "="*80)
print("STARTING TRAINING NOW - You will see live updates below")
print("="*80)
print("\nProgress will show:")
print("  - Training loss decreasing over time")
print("  - Evaluation metrics after each epoch")
print("  - Checkpoint saves")
print("\nPress Ctrl+C to stop (checkpoints will be saved)")
print("="*80 + "\n")

print("\n" + "="*80)
print("TRAINING STARTED")
print("="*80)
print("You can press Ctrl+C to stop (checkpoints will be saved)")

try:
    print("\n[INFO] Training starting...")
    print("[INFO] You will see progress updates every 10 steps")
    print("[INFO] Each epoch will show evaluation results\n")
    
    trainer.train()
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    print("[INFO] Running final evaluation on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"\nTest set results:")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save final model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    print("[INFO] Saving model weights...")
    trainer.save_model(str(model_output_dir))
    print("[INFO] Saving tokenizer...")
    tokenizer.save_pretrained(str(model_output_dir))
    print("[INFO] Model files saved!")
    
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
            "epochs": training_args.num_train_epochs,
        }
    }
    
    info_file = model_output_dir / "training_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Model saved to: {model_output_dir}")
    print("\nSaved files:")
    print("  - Model weights (pytorch_model.bin or model.safetensors)")
    print("  - Tokenizer files")
    print("  - Config (config.json)")
    print("  - Training info (training_info.json)")
    print("  - Checkpoints (checkpoint-1/, checkpoint-2/, checkpoint-3/)")
    
    print("\n" + "="*80)
    print("MODEL READY FOR REUSE")
    print("="*80)
    print("\nYou can now load and use this model multiple times!")
    print("\nTo load the model:")
    print(f"  from transformers import AutoModelForSequenceClassification, AutoTokenizer")
    print(f"  model = AutoModelForSequenceClassification.from_pretrained(r'{model_output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained(r'{model_output_dir}')")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")
    print(f"\nCheckpoints saved in: {model_output_dir}")
    print("You can:")
    print("  1. Resume training from the latest checkpoint")
    print("  2. Use the latest checkpoint as your model")
    print(f"  3. Load from: {model_output_dir}/checkpoint-*/")
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

