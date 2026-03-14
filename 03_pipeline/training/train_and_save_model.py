"""
Train model and save locally for reuse.

This script trains the model on your final training set and saves it
so you can use it multiple times without retraining.
"""

import subprocess
import sys
from pathlib import Path

print("="*80)
print("TRAIN AND SAVE MODEL LOCALLY")
print("="*80)

# Paths
project_root = Path(__file__).parent.parent.parent
training_script = project_root / "modelB_training.py"
final_training_data = Path(r"F:\DATA STORAGE\AGG_PACKET\final_training_set\train.json")
model_output_dir = project_root / "saved_models" / "depression_classifier"

# Check if files exist
if not training_script.exists():
    print(f"ERROR: Training script not found: {training_script}")
    sys.exit(1)

if not final_training_data.exists():
    print(f"ERROR: Training data not found: {final_training_data}")
    sys.exit(1)

print(f"\nTraining script: {training_script}")
print(f"Training data: {final_training_data}")
print(f"Model will be saved to: {model_output_dir}")

# Create output directory
model_output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n[OK] Output directory ready: {model_output_dir}")

print("\n" + "="*80)
print("TRAINING CONFIGURATION")
print("="*80)
print("""
Model: rafalposwiata/deproberta-large-depression (pretrained)
Training data: Final training set with corrected labels
  - Label 1 = Depression
  - Label 0 = Non-depression

Training parameters:
  - Learning rate: 2e-5
  - Batch size: 4 per device
  - Epochs: 3
  - Evaluation: After each epoch
  - Best model saved automatically

Model will be saved to: saved_models/depression_classifier/
You can reload and use this model multiple times without retraining.
""")

response = input("\nStart training? (y/n): ").strip().lower()
if response != 'y':
    print("Cancelled.")
    sys.exit(0)

# Build command
# Note: We need to use all_labeled_users.json or point to the directory
# The training script expects user records with 'label' field
training_data_source = Path(r"F:\DATA STORAGE\AGG_PACKET\labeling_outputs\all_labeled_users.json")

cmd = [
    sys.executable,
    str(training_script),
    "--dataset-path", str(training_data_source),
    "--label-field", "label",
    "--train",
    "--auto",
    "--max-users", "0",  # Use all data
    "--output-dir", str(model_output_dir)
]

print("\n" + "="*80)
print("STARTING TRAINING...")
print("="*80)
print("This will save the model to:", model_output_dir)
print("You can use this saved model multiple times without retraining.\n")

try:
    subprocess.run(cmd, check=True, cwd=str(project_root))
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {model_output_dir}")
    print("\nTo use the saved model later:")
    print(f"  from transformers import AutoModelForSequenceClassification, AutoTokenizer")
    print(f"  model = AutoModelForSequenceClassification.from_pretrained(r'{model_output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained(r'{model_output_dir}')")
    
except subprocess.CalledProcessError as e:
    print(f"\nERROR: Training failed with exit code {e.returncode}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")
    print("Partial model may be saved in:", model_output_dir)
    sys.exit(1)




