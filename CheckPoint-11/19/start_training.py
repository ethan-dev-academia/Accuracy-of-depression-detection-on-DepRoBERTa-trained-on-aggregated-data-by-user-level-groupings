"""
Quick script to start training with your labeled data.
This script prepares and runs the training command.
"""

import subprocess
import sys
from pathlib import Path

print("="*80)
print("STARTING ML MODEL TRAINING")
print("="*80)

# Paths
project_root = Path(__file__).parent.parent.parent
training_script = project_root / "modelB_training.py"
training_data = Path(r"F:\DATA STORAGE\AGG_PACKET\labeling_outputs\all_labeled_users.json")

# Check if files exist
if not training_script.exists():
    print(f"ERROR: Training script not found: {training_script}")
    sys.exit(1)

if not training_data.exists():
    print(f"ERROR: Training data not found: {training_data}")
    print("\nAvailable files in labeling_outputs:")
    output_dir = training_data.parent
    if output_dir.exists():
        for f in output_dir.glob("*.json"):
            print(f"  - {f.name}")
    sys.exit(1)

print(f"\nTraining script: {training_script}")
print(f"Training data: {training_data}")
print(f"Data size: {training_data.stat().st_size / (1024*1024):.1f} MB")

# Build command
cmd = [
    sys.executable,
    str(training_script),
    "--dataset-path", str(training_data),
    "--label-field", "label",
    "--train",
    "--auto",
    "--max-users", "0",  # Use all data
    "--output-dir", "./ModelB_final"
]

print("\n" + "="*80)
print("TRAINING COMMAND:")
print("="*80)
print(" ".join(cmd))

print("\n" + "="*80)
print("TRAINING PARAMETERS:")
print("="*80)
print("""
- Model: rafalposwiata/deproberta-large-depression
- Learning rate: 2e-5
- Batch size: 4 per device
- Epochs: 3
- Evaluation: After each epoch
- Metric: F1 score
""")

response = input("\nStart training? (y/n): ").strip().lower()
if response != 'y':
    print("Cancelled.")
    sys.exit(0)

print("\n" + "="*80)
print("STARTING TRAINING...")
print("="*80)
print("This may take several hours. Monitor the output for progress.\n")

# Run training
try:
    subprocess.run(cmd, check=True, cwd=str(project_root))
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("Model saved to: ./ModelB_final")
except subprocess.CalledProcessError as e:
    print(f"\nERROR: Training failed with exit code {e.returncode}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")
    sys.exit(1)

