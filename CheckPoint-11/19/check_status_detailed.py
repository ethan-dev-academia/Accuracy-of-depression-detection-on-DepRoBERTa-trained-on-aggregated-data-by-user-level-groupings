"""Detailed training status check"""
import json
from pathlib import Path
import os
import subprocess
import sys

model_dir = Path(r"F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\saved_models\depression_classifier_final")

print("="*80)
print("DETAILED TRAINING STATUS")
print("="*80)

print(f"\nDirectory: {model_dir}")
print(f"Exists: {model_dir.exists()}")

if not model_dir.exists():
    print("\n[STATUS] Training directory not created yet.")
    print("Training may not have started, or it's still initializing.")
    sys.exit(0)

# List all files
all_items = list(model_dir.rglob('*'))
print(f"\nTotal items in directory: {len(all_items)}")

# Check for checkpoints
checkpoints = [f for f in all_items if 'checkpoint' in f.name and f.is_dir()]
print(f"\nCheckpoints found: {len(checkpoints)}")
if checkpoints:
    checkpoint_dirs = sorted(set([f.parent if f.is_file() else f for f in checkpoints]))
    for ckpt in checkpoint_dirs:
        if ckpt.is_dir():
            print(f"  - {ckpt.name}")

# Check for log files
log_files = [f for f in all_items if 'log' in f.name.lower() or 'runs' in str(f)]
print(f"\nLog files/directories: {len(log_files)}")
if log_files:
    for log in log_files[:5]:  # Show first 5
        print(f"  - {log.name}")

# Check for config
config_file = model_dir / "config.json"
print(f"\nConfig file: {'EXISTS' if config_file.exists() else 'NOT FOUND'}")

# Check for training info (means training completed)
info_file = model_dir / "training_info.json"
print(f"Training info: {'EXISTS (TRAINING COMPLETED!)' if info_file.exists() else 'NOT FOUND (Still training or not started)'}")

if info_file.exists():
    with open(info_file, 'r', encoding='utf-8') as f:
        info = json.load(f)
    print("\n" + "="*80)
    print("TRAINING RESULTS")
    print("="*80)
    print(f"Test Accuracy: {info.get('test_results', {}).get('eval_accuracy', 'N/A')}")
    print(f"Test F1 Score: {info.get('test_results', {}).get('eval_f1', 'N/A')}")
    print(f"Training Examples: {info.get('training_examples', 'N/A'):,}")
    print(f"Validation Examples: {info.get('validation_examples', 'N/A'):,}")
    print(f"Test Examples: {info.get('test_examples', 'N/A'):,}")

# Check for model files
model_files = [f for f in all_items if any(ext in f.name for ext in ['.bin', '.safetensors', 'pytorch_model'])]
print(f"\nModel weight files: {len(model_files)}")
if model_files:
    for mf in model_files[:3]:
        size_mb = mf.stat().st_size / (1024*1024) if mf.is_file() else 0
        print(f"  - {mf.name} ({size_mb:.1f} MB)")

# Check if Python process is running
print("\n" + "="*80)
print("PROCESS CHECK")
print("="*80)
try:
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                          capture_output=True, text=True, shell=True)
    if 'python.exe' in result.stdout:
        print("Python processes running:")
        lines = [l for l in result.stdout.split('\n') if 'python.exe' in l]
        for line in lines[:5]:
            print(f"  {line.strip()}")
    else:
        print("No Python processes found (training may have completed or not started)")
except:
    print("Could not check running processes")

print("\n" + "="*80)




