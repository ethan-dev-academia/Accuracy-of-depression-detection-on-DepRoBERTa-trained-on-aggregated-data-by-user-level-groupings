"""
Run training with live terminal output.
This script runs the training in the foreground so you can see all updates.
"""
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).parent
training_script = script_dir / "train_final_model.py"

print("="*80)
print("STARTING TRAINING WITH LIVE UPDATES")
print("="*80)
print(f"\nRunning: {training_script}")
print("\nYou will see:")
print("  - Model download progress (if needed)")
print("  - Tokenization progress")
print("  - Training loss updates every 10 steps")
print("  - Evaluation results after each epoch")
print("  - Checkpoint saves")
print("\nPress Ctrl+C to stop (checkpoints will be saved)")
print("="*80 + "\n")

# Run the training script in the foreground
try:
    subprocess.run([sys.executable, str(training_script)], check=True)
except KeyboardInterrupt:
    print("\n\n[INFO] Training interrupted by user")
    print("[INFO] Checkpoints are saved - you can resume later")
except subprocess.CalledProcessError as e:
    print(f"\n[ERROR] Training failed with exit code {e.returncode}")
    sys.exit(1)




