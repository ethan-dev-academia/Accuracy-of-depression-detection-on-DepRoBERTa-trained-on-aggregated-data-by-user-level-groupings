"""
Quick script to check training status and progress.
"""
import json
from pathlib import Path
import time

model_dir = Path(__file__).parent.parent.parent / "saved_models" / "depression_classifier_final"

print("="*80)
print("TRAINING STATUS CHECK")
print("="*80)

if not model_dir.exists():
    print(f"\n[INFO] Training directory not found yet: {model_dir}")
    print("This means training hasn't started or just started.")
    print("Training may take several hours on CPU.")
else:
    print(f"\n[FOUND] Training directory: {model_dir}")
    
    # Check for checkpoints
    checkpoints = list(model_dir.glob("checkpoint-*"))
    if checkpoints:
        print(f"\n[CHECKPOINTS] Found {len(checkpoints)} checkpoint(s):")
        for ckpt in sorted(checkpoints):
            print(f"  - {ckpt.name}")
    
    # Check for logs
    log_dir = model_dir / "logs"
    if log_dir.exists():
        print(f"\n[LOGS] Log directory exists: {log_dir}")
    
    # Check for final model
    config_file = model_dir / "config.json"
    if config_file.exists():
        print(f"\n[MODEL] Model files found!")
        with open(config_file, 'r') as f:
            config = json.load(f)
            print(f"  Labels: {config.get('id2label', 'N/A')}")
    
    # Check for training info
    info_file = model_dir / "training_info.json"
    if info_file.exists():
        print(f"\n[TRAINING INFO] Training completed!")
        with open(info_file, 'r') as f:
            info = json.load(f)
            print(f"  Test accuracy: {info.get('test_results', {}).get('eval_accuracy', 'N/A')}")
            print(f"  Test F1: {info.get('test_results', {}).get('eval_f1', 'N/A')}")

print("\n" + "="*80)
print("To monitor training progress:")
print(f"  1. Check logs in: {model_dir / 'logs'}")
print(f"  2. Look for checkpoints in: {model_dir}")
print(f"  3. Run this script again to see updates")
print("="*80)

