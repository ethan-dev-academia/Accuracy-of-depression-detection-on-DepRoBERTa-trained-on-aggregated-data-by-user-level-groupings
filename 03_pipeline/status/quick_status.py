"""Quick training status"""
from pathlib import Path
import time

model_dir = Path(r"F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\saved_models\depression_classifier_final")
final_data = Path(r"F:\DATA STORAGE\AGG_PACKET\final_training_set\train.json")

print("="*80)
print("TRAINING STATUS SUMMARY")
print("="*80)

# Check if training directory exists
if model_dir.exists():
    items = list(model_dir.rglob('*'))
    file_count = len([f for f in items if f.is_file()])
    dir_count = len([f for f in items if f.is_dir()])
    
    print(f"\n[OK] Training directory exists: {model_dir}")
    print(f"  Files: {file_count}, Directories: {dir_count}")
    
    if file_count == 0 and dir_count == 0:
        print("\n[STATUS] Initialization phase")
        print("  - Directory created but no files yet")
        print("  - Training script is likely:")
        print("    • Downloading the pretrained model (first time only)")
        print("    • Tokenizing 27K+ training examples")
        print("    • Setting up training environment")
        print("\n  This can take 10-30 minutes before training actually starts.")
        print("  Please wait...")
    else:
        checkpoints = [f for f in items if 'checkpoint' in f.name]
        if checkpoints:
            print(f"\n[STATUS] Training in progress!")
            print(f"  Found {len(checkpoints)} checkpoint(s)")
        else:
            print("\n[STATUS] Training may have just started")
else:
    print(f"\n[NOT FOUND] Training directory not found")
    print("  Training may not have started yet")

# Check if data exists
if final_data.exists():
    print(f"\n[OK] Training data ready: {final_data}")
else:
    print(f"\n[NOT FOUND] Training data not found: {final_data}")

print("\n" + "="*80)
print("WHAT TO EXPECT:")
print("="*80)
print("1. Initialization: 10-30 minutes (downloading model, tokenizing data)")
print("2. Training: 6-12 hours on CPU (3 epochs)")
print("3. Checkpoints: Saved after each epoch")
print("4. Final model: Saved when training completes")
print("\nCheck status again in a few minutes to see progress!")
print("="*80)

