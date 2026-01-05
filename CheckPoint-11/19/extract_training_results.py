"""
Extract training results from the saved model and create a summary file.
"""
import json
from pathlib import Path
from datetime import datetime

model_dir = Path(r"F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\saved_models\depression_classifier_final")
output_file = Path(__file__).parent / "training_results_summary.txt"

print("Extracting training results...")

results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_directory": str(model_dir),
}

# Check for training info
info_file = model_dir / "training_info.json"
if info_file.exists():
    with open(info_file, 'r', encoding='utf-8') as f:
        info = json.load(f)
    results.update({
        "model_name": info.get("model_name", "N/A"),
        "training_examples": info.get("training_examples", "N/A"),
        "validation_examples": info.get("validation_examples", "N/A"),
        "test_examples": info.get("test_examples", "N/A"),
        "test_results": info.get("test_results", {}),
        "label_mapping": info.get("label_mapping", {}),
        "training_args": info.get("training_args", {}),
    })
    print("Found training info file!")
else:
    print("Training info file not found")

# Write summary to file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("TRAINING RESULTS SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Extracted: {results['timestamp']}\n")
    f.write(f"Model Directory: {results['model_directory']}\n\n")
    
    if "model_name" in results:
        f.write("="*80 + "\n")
        f.write("MODEL INFORMATION\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {results['model_name']}\n")
        f.write(f"Label Mapping: {results.get('label_mapping', {})}\n\n")
        
        f.write("="*80 + "\n")
        f.write("DATASET INFORMATION\n")
        f.write("="*80 + "\n")
        f.write(f"Training Examples: {results.get('training_examples', 'N/A'):,}\n")
        f.write(f"Validation Examples: {results.get('validation_examples', 'N/A'):,}\n")
        f.write(f"Test Examples: {results.get('test_examples', 'N/A'):,}\n\n")
        
        f.write("="*80 + "\n")
        f.write("TEST SET RESULTS\n")
        f.write("="*80 + "\n")
        test_results = results.get('test_results', {})
        for key, value in test_results.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("TRAINING CONFIGURATION\n")
        f.write("="*80 + "\n")
        train_args = results.get('training_args', {})
        for key, value in train_args.items():
            f.write(f"{key}: {value}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("END OF SUMMARY\n")
    f.write("="*80 + "\n")

print(f"\nResults summary saved to: {output_file}")

