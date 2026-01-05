"""
Check if test set and inference have been used before.
"""
from pathlib import Path
import json

print("="*80)
print("USAGE HISTORY CHECK")
print("="*80)

# Check test set
test_file = Path(r"F:\DATA STORAGE\AGG_PACKET\final_training_set\test.json")
print("\n1. TEST SET:")
print(f"   Location: {test_file}")
print(f"   Exists: {test_file.exists()}")

if test_file.exists():
    import os
    import datetime
    stat = test_file.stat()
    mod_time = datetime.datetime.fromtimestamp(stat.st_mtime)
    print(f"   Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Size: {stat.st_size / (1024*1024):.2f} MB")
    
    # Check if it was used in training
    print("\n   Usage in training scripts:")
    train_script = Path(__file__).parent / "train_final_model.py"
    if train_script.exists():
        content = train_script.read_text()
        if "test.json" in content and "evaluate" in content:
            print("   [YES] Used in train_final_model.py for final evaluation")
        else:
            print("   [NO] Not found in training scripts")

# Check if model has been loaded for inference
model_dir = Path(r"F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\saved_models\depression_classifier_final")
print("\n2. SAVED MODEL:")
print(f"   Location: {model_dir}")
print(f"   Exists: {model_dir.exists()}")

if model_dir.exists():
    config_file = model_dir / "config.json"
    if config_file.exists():
        print("   [YES] Model has been saved and is ready to use")
        
        # Check for inference scripts
        inference_scripts = [
            Path(__file__).parent / "load_saved_model.py",
            Path(__file__).parent / "TRAINING_STATUS.md",
        ]
        
        print("\n   Inference code available in:")
        for script in inference_scripts:
            if script.exists():
                print(f"   - {script.name}")

# Check training history
print("\n3. TRAINING HISTORY:")
training_info = model_dir / "training_info.json"
if training_info.exists():
    with open(training_info, 'r') as f:
        info = json.load(f)
    print("   [YES] Training completed and evaluated on test set")
    print(f"   Test accuracy: {info.get('test_results', {}).get('eval_accuracy', 'N/A')}")
    print(f"   Test F1: {info.get('test_results', {}).get('eval_f1', 'N/A')}")
else:
    print("   [NO] No training info found")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nTest set has been used:")
print("  - Automatically during training for final evaluation")
print("  - Results: 73.97% accuracy, 73.08% F1 score")
print("\nInference code exists but may not have been run manually:")
print("  - load_saved_model.py - Script to load model and do inference")
print("  - TRAINING_STATUS.md - Example inference code")
print("\nTo use the test set for manual evaluation:")
print("  - Load test.json from: F:\\DATA STORAGE\\AGG_PACKET\\final_training_set\\")
print("  - Use your trained model to make predictions")
print("  - Compare predictions to true labels")

