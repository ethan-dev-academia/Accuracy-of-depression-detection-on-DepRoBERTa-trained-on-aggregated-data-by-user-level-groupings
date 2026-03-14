"""
Compare your trained model with the default DepRoBERTa model.

This script:
1. Loads both models (your trained model and default DepRoBERTa)
2. Runs predictions on the test set with both models
3. Calculates comprehensive metrics
4. Generates raw data tables with all results
"""

import json
import csv
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm
import sys

print("="*80)
print("MODEL COMPARISON: Your Model vs Default DepRoBERTa")
print("="*80)

# Paths
test_file = Path(r"F:\DATA STORAGE\AGG_PACKET\final_training_set\test.json")
your_model_dir = Path(r"F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\saved_models\depression_classifier_final")
default_model_name = "rafalposwiata/deproberta-large-depression"
output_dir = Path(__file__).parent / "model_comparison_results"
output_dir.mkdir(exist_ok=True)

print(f"\nTest set: {test_file}")
print(f"Your model: {your_model_dir}")
print(f"Default model: {default_model_name}")
print(f"Output directory: {output_dir}")

# Load test set
print("\n" + "="*80)
print("LOADING TEST SET")
print("="*80)
with open(test_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data):,} test examples")

# Extract true labels and texts
true_labels = [record['label'] for record in test_data]
texts = [record['text'] for record in test_data]

print(f"Label distribution:")
label_counts = Counter(true_labels)
for label, count in sorted(label_counts.items()):
    label_name = {0: "non-depression", 1: "depression"}.get(label, f"unknown({label})")
    print(f"  {label_name}: {count:,} ({count/len(test_data)*100:.1f}%)")

# Load your trained model
print("\n" + "="*80)
print("LOADING YOUR TRAINED MODEL")
print("="*80)
print("Loading tokenizer...")
your_tokenizer = AutoTokenizer.from_pretrained(str(your_model_dir))
print("Loading model...")
your_model = AutoModelForSequenceClassification.from_pretrained(str(your_model_dir))
your_model.eval()
print("[OK] Your model loaded successfully!")
print(f"  Labels: {your_model.config.id2label}")

# Load default DepRoBERTa model
print("\n" + "="*80)
print("LOADING DEFAULT DEPROBERTA MODEL")
print("="*80)
print("Loading tokenizer...")
default_tokenizer = AutoTokenizer.from_pretrained(default_model_name)
print("Loading model...")
default_model = AutoModelForSequenceClassification.from_pretrained(default_model_name)
default_model.eval()
print("[OK] Default model loaded successfully!")
print(f"  Labels: {default_model.config.id2label}")

# Run predictions
print("\n" + "="*80)
print("RUNNING PREDICTIONS")
print("="*80)
print("This may take 30-60 minutes on CPU...")
print("Processing test examples...\n")

your_predictions = []
your_confidences = []
default_predictions_3class = []
default_confidences_3class = []
default_predictions_binary = []
default_confidences_binary = []

batch_size = 8  # Process in batches for efficiency

for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
    batch_texts = texts[i:i+batch_size]
    
    # Your model predictions
    your_inputs = your_tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    with torch.no_grad():
        your_outputs = your_model(**your_inputs)
        your_probs = torch.nn.functional.softmax(your_outputs.logits, dim=-1)
        your_batch_preds = torch.argmax(your_probs, dim=-1).cpu().numpy()
        your_batch_confs = torch.max(your_probs, dim=-1)[0].cpu().numpy()
    
    your_predictions.extend(your_batch_preds.tolist())
    your_confidences.extend(your_batch_confs.tolist())
    
    # Default model predictions
    default_inputs = default_tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    with torch.no_grad():
        default_outputs = default_model(**default_inputs)
        default_probs = torch.nn.functional.softmax(default_outputs.logits, dim=-1)
        default_batch_preds_3class = torch.argmax(default_probs, dim=-1).cpu().numpy()
        default_batch_confs_3class = torch.max(default_probs, dim=-1)[0].cpu().numpy()
    
    # Map 3-class to binary: 0="not depression" -> 0, 1="moderate" and 2="severe" -> 1
    default_batch_preds_binary = (default_batch_preds_3class > 0).astype(int)
    # For binary confidence, take max of moderate+severe probabilities
    default_batch_confs_binary = (
        default_probs[:, 1] + default_probs[:, 2]
    ).cpu().numpy()
    
    default_predictions_3class.extend(default_batch_preds_3class.tolist())
    default_confidences_3class.extend(default_batch_confs_3class.tolist())
    default_predictions_binary.extend(default_batch_preds_binary.tolist())
    default_confidences_binary.extend(default_batch_confs_binary.tolist())

print("\n[OK] Predictions complete!")

# Calculate metrics
print("\n" + "="*80)
print("CALCULATING METRICS")
print("="*80)

# Your model metrics
your_accuracy = accuracy_score(true_labels, your_predictions)
your_f1 = f1_score(true_labels, your_predictions, average='weighted')
your_precision = precision_score(true_labels, your_predictions, average='weighted', zero_division=0)
your_recall = recall_score(true_labels, your_predictions, average='weighted', zero_division=0)
your_cm = confusion_matrix(true_labels, your_predictions)

# Default model (3-class) metrics
default_3class_accuracy = accuracy_score(true_labels, default_predictions_binary)  # Compare binary mapped
default_3class_f1 = f1_score(true_labels, default_predictions_binary, average='weighted')
default_3class_precision = precision_score(true_labels, default_predictions_binary, average='weighted', zero_division=0)
default_3class_recall = recall_score(true_labels, default_predictions_binary, average='weighted', zero_division=0)
default_3class_cm = confusion_matrix(true_labels, default_predictions_binary)

# Default model (binary mapped) - same as above but for clarity
default_binary_accuracy = default_3class_accuracy
default_binary_f1 = default_3class_f1
default_binary_precision = default_3class_precision
default_binary_recall = default_3class_recall
default_binary_cm = default_3class_cm

# 3-class distribution from default model
default_3class_distribution = Counter(default_predictions_3class)
default_3class_labels = {0: "not depression", 1: "moderate", 2: "severe"}

print("\nYour Model Metrics:")
print(f"  Accuracy: {your_accuracy:.4f} ({your_accuracy*100:.2f}%)")
print(f"  F1 Score: {your_f1:.4f} ({your_f1*100:.2f}%)")
print(f"  Precision: {your_precision:.4f} ({your_precision*100:.2f}%)")
print(f"  Recall: {your_recall:.4f} ({your_recall*100:.2f}%)")

print("\nDefault Model (Binary Mapped) Metrics:")
print(f"  Accuracy: {default_binary_accuracy:.4f} ({default_binary_accuracy*100:.2f}%)")
print(f"  F1 Score: {default_binary_f1:.4f} ({default_binary_f1*100:.2f}%)")
print(f"  Precision: {default_binary_precision:.4f} ({default_binary_precision*100:.2f}%)")
print(f"  Recall: {default_binary_recall:.4f} ({default_binary_recall*100:.2f}%)")

print("\nDefault Model 3-Class Distribution:")
for label_id, count in sorted(default_3class_distribution.items()):
    label_name = default_3class_labels.get(label_id, f"unknown({label_id})")
    print(f"  {label_name}: {count:,} ({count/len(test_data)*100:.1f}%)")

# Calculate improvements
accuracy_improvement = your_accuracy - default_binary_accuracy
f1_improvement = your_f1 - default_binary_f1

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print(f"\nAccuracy Improvement: {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.2f}%)")
print(f"F1 Score Improvement: {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)")

if accuracy_improvement > 0:
    print("\n[SUCCESS] Your model performs BETTER than default!")
elif accuracy_improvement < 0:
    print("\n[INFO] Default model performs better on this test set.")
else:
    print("\n[INFO] Models perform similarly.")

# Generate per-example results
print("\n" + "="*80)
print("GENERATING DATA TABLES")
print("="*80)

# Per-example predictions table
per_example_file = output_dir / "per_example_predictions.csv"
print(f"\nCreating per-example predictions table: {per_example_file.name}")

with open(per_example_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'example_id',
        'text_preview',
        'true_label',
        'your_prediction',
        'your_confidence',
        'default_3class_prediction',
        'default_3class_label',
        'default_3class_confidence',
        'default_binary_prediction',
        'default_binary_confidence',
        'models_agree',
        'your_correct',
        'default_correct'
    ])
    
    for i, (text, true_label, your_pred, your_conf, 
            default_3pred, default_3conf, default_bpred, default_bconf) in enumerate(zip(
        texts, true_labels, your_predictions, your_confidences,
        default_predictions_3class, default_confidences_3class,
        default_predictions_binary, default_confidences_binary
    )):
        text_preview = text[:200] + "..." if len(text) > 200 else text
        default_3label = default_3class_labels.get(default_3pred, f"unknown({default_3pred})")
        models_agree = "Yes" if your_pred == default_bpred else "No"
        your_correct = "Yes" if your_pred == true_label else "No"
        default_correct = "Yes" if default_bpred == true_label else "No"
        
        writer.writerow([
            i,
            text_preview,
            true_label,
            your_pred,
            f"{your_conf:.4f}",
            default_3pred,
            default_3label,
            f"{default_3conf:.4f}",
            default_bpred,
            f"{default_bconf:.4f}",
            models_agree,
            your_correct,
            default_correct
        ])

print(f"  [OK] Created {len(test_data):,} rows")

# Summary comparison table
summary_file = output_dir / "comparison_results_summary.json"
print(f"\nCreating summary comparison: {summary_file.name}")

summary_data = {
    "your_model": {
        "accuracy": float(your_accuracy),
        "f1_score": float(your_f1),
        "precision": float(your_precision),
        "recall": float(your_recall),
        "confusion_matrix": your_cm.tolist()
    },
    "default_model_3class": {
        "distribution": {str(k): int(v) for k, v in sorted(default_3class_distribution.items())},
        "label_mapping": default_3class_labels
    },
    "default_model_binary": {
        "accuracy": float(default_binary_accuracy),
        "f1_score": float(default_binary_f1),
        "precision": float(default_binary_precision),
        "recall": float(default_binary_recall),
        "confusion_matrix": default_binary_cm.tolist()
    },
    "comparison": {
        "accuracy_improvement": float(accuracy_improvement),
        "f1_improvement": float(f1_improvement),
        "accuracy_improvement_percent": float(accuracy_improvement * 100),
        "f1_improvement_percent": float(f1_improvement * 100)
    },
    "test_set_info": {
        "total_examples": len(test_data),
        "label_distribution": {str(k): int(v) for k, v in sorted(label_counts.items())}
    }
}

with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)

print("  [OK] Summary saved")

# Confusion matrices
confusion_file = output_dir / "confusion_matrices.json"
print(f"\nCreating confusion matrices: {confusion_file.name}")

confusion_data = {
    "your_model": {
        "matrix": your_cm.tolist(),
        "labels": ["non-depression", "depression"]
    },
    "default_model_binary": {
        "matrix": default_binary_cm.tolist(),
        "labels": ["non-depression", "depression"]
    }
}

with open(confusion_file, 'w', encoding='utf-8') as f:
    json.dump(confusion_data, f, indent=2, ensure_ascii=False)

print("  [OK] Confusion matrices saved")

# Disagreement analysis
disagreement_file = output_dir / "disagreement_analysis.csv"
print(f"\nCreating disagreement analysis: {disagreement_file.name}")

disagreements = []
for i, (text, true_label, your_pred, your_conf, default_bpred, default_bconf) in enumerate(zip(
    texts, true_labels, your_predictions, your_confidences,
    default_predictions_binary, default_confidences_binary
)):
    if your_pred != default_bpred:
        disagreements.append({
            'example_id': i,
            'text_preview': text[:300] + "..." if len(text) > 300 else text,
            'true_label': true_label,
            'your_prediction': your_pred,
            'your_confidence': your_conf,
            'default_prediction': default_bpred,
            'default_confidence': default_bconf,
            'your_correct': your_pred == true_label,
            'default_correct': default_bpred == true_label
        })

with open(disagreement_file, 'w', newline='', encoding='utf-8') as f:
    if disagreements:
        writer = csv.DictWriter(f, fieldnames=disagreements[0].keys())
        writer.writeheader()
        writer.writerows(disagreements)
    else:
        writer = csv.writer(f)
        writer.writerow(['No disagreements found - models agree on all examples'])

print(f"  [OK] Found {len(disagreements)} disagreements ({len(disagreements)/len(test_data)*100:.1f}%)")

# Class distribution analysis
distribution_file = output_dir / "class_distribution_analysis.json"
print(f"\nCreating class distribution analysis: {distribution_file.name}")

distribution_data = {
    "default_model_3class_distribution": {
        str(k): {
            "count": int(v),
            "percentage": float(v / len(test_data) * 100),
            "label": default_3class_labels.get(k, f"unknown({k})")
        }
        for k, v in sorted(default_3class_distribution.items())
    },
    "mapping_to_binary": {
        "not_depression (0)": "maps to non-depression (0)",
        "moderate (1)": "maps to depression (1)",
        "severe (2)": "maps to depression (1)"
    }
}

with open(distribution_file, 'w', encoding='utf-8') as f:
    json.dump(distribution_data, f, indent=2, ensure_ascii=False)

print("  [OK] Distribution analysis saved")

# Generate comparison report
report_file = output_dir / "comparison_report.md"
print(f"\nCreating comparison report: {report_file.name}")

with open(report_file, 'w', encoding='utf-8') as f:
    f.write("# Model Comparison Report\n\n")
    f.write("## Your Trained Model vs Default DepRoBERTa\n\n")
    f.write(f"**Test Set**: {len(test_data):,} examples\n\n")
    
    f.write("## Summary Metrics\n\n")
    f.write("| Metric | Your Model | Default Model (Binary) | Improvement |\n")
    f.write("|--------|------------|------------------------|-------------|\n")
    f.write(f"| Accuracy | {your_accuracy:.4f} ({your_accuracy*100:.2f}%) | {default_binary_accuracy:.4f} ({default_binary_accuracy*100:.2f}%) | {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.2f}%) |\n")
    f.write(f"| F1 Score | {your_f1:.4f} ({your_f1*100:.2f}%) | {default_binary_f1:.4f} ({default_binary_f1*100:.2f}%) | {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%) |\n")
    f.write(f"| Precision | {your_precision:.4f} ({your_precision*100:.2f}%) | {default_binary_precision:.4f} ({default_binary_precision*100:.2f}%) | {your_precision - default_binary_precision:+.4f} |\n")
    f.write(f"| Recall | {your_recall:.4f} ({your_recall*100:.2f}%) | {default_binary_recall:.4f} ({default_binary_recall*100:.2f}%) | {your_recall - default_binary_recall:+.4f} |\n\n")
    
    f.write("## Default Model 3-Class Distribution\n\n")
    for label_id, count in sorted(default_3class_distribution.items()):
        label_name = default_3class_labels.get(label_id, f"unknown({label_id})")
        f.write(f"- **{label_name}**: {count:,} ({count/len(test_data)*100:.1f}%)\n")
    f.write("\n")
    
    f.write("## Disagreement Analysis\n\n")
    f.write(f"- **Total disagreements**: {len(disagreements)} ({len(disagreements)/len(test_data)*100:.1f}%)\n")
    f.write(f"- **Agreement rate**: {(1 - len(disagreements)/len(test_data))*100:.1f}%\n\n")
    
    f.write("## Conclusion\n\n")
    if accuracy_improvement > 0:
        f.write("✅ **Your trained model performs BETTER than the default DepRoBERTa model** on this test set.\n\n")
        f.write(f"Your fine-tuning improved accuracy by {accuracy_improvement*100:.2f}% and F1 score by {f1_improvement*100:.2f}%.\n")
    elif accuracy_improvement < 0:
        f.write("ℹ️ **The default DepRoBERTa model performs better** on this test set.\n\n")
        f.write(f"The default model has {abs(accuracy_improvement*100):.2f}% higher accuracy.\n")
    else:
        f.write("ℹ️ **Both models perform similarly** on this test set.\n\n")
    
    f.write("\n## Files Generated\n\n")
    f.write("- `per_example_predictions.csv` - Full test set with all predictions\n")
    f.write("- `comparison_results_summary.json` - Overall metrics\n")
    f.write("- `confusion_matrices.json` - Confusion matrices for both models\n")
    f.write("- `disagreement_analysis.csv` - Cases where models disagree\n")
    f.write("- `class_distribution_analysis.json` - 3-class distribution from default model\n")

print("  [OK] Report generated")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
print(f"\nAll results saved to: {output_dir}")
print("\nGenerated files:")
print(f"  1. per_example_predictions.csv ({len(test_data):,} rows)")
print(f"  2. comparison_results_summary.json")
print(f"  3. confusion_matrices.json")
print(f"  4. disagreement_analysis.csv ({len(disagreements)} disagreements)")
print(f"  5. class_distribution_analysis.json")
print(f"  6. comparison_report.md")
print("\n" + "="*80)




