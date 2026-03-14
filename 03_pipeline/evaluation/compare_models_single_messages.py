"""
Compare your trained model with default DepRoBERTa on individual messages.

This script:
1. Loads labeled user data with individual posts and comments
2. Extracts individual messages (posts/comments) as separate test examples
3. Tests both models on each individual message
4. Compares accuracy between models
5. Records results
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
print("MODEL COMPARISON: Individual Messages (Single Posts/Comments)")
print("="*80)

# Paths
labeled_data_file = Path(r"F:\DATA STORAGE\AGG_PACKET\labeling_outputs\all_labeled_users.json")
your_model_dir = Path(r"F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\saved_models\depression_classifier_final")
default_model_name = "rafalposwiata/deproberta-large-depression"
output_dir = Path(__file__).parent / "single_message_comparison_results"
output_dir.mkdir(exist_ok=True)

print(f"\nLabeled data: {labeled_data_file}")
print(f"Your model: {your_model_dir}")
print(f"Default model: {default_model_name}")
print(f"Output directory: {output_dir}")

# Load labeled user data
print("\n" + "="*80)
print("LOADING LABELED USER DATA")
print("="*80)

if not labeled_data_file.exists():
    print(f"ERROR: Labeled data file not found: {labeled_data_file}")
    print("\nTrying alternative location...")
    alt_file = Path(r"F:\DATA STORAGE\AGG_PACKET") / "all_labeled_users.json"
    if alt_file.exists():
        labeled_data_file = alt_file
        print(f"Found at: {labeled_data_file}")
    else:
        # Try to find any labeled file
        agg_packet_dir = Path(r"F:\DATA STORAGE\AGG_PACKET")
        labeled_files = list(agg_packet_dir.glob("reddit_user_analysis_*_labeled.json"))
        if labeled_files:
            labeled_data_file = labeled_files[0]
            print(f"Found labeled file: {labeled_data_file}")
        else:
            print("ERROR: No labeled data files found!")
            sys.exit(1)

with open(labeled_data_file, 'r', encoding='utf-8') as f:
    user_data = json.load(f)

print(f"Loaded {len(user_data):,} labeled users")

# Extract individual messages
print("\n" + "="*80)
print("EXTRACTING INDIVIDUAL MESSAGES")
print("="*80)

def is_valid_text(text):
    """Check if text is valid (not removed/deleted/empty)."""
    if not text or not isinstance(text, str):
        return False
    text_lower = text.strip().lower()
    return text_lower not in ["[removed]", "[deleted]", ""] and len(text.strip()) >= 10

individual_messages = []
message_labels = []
message_types = []  # 'post' or 'comment'
message_sources = []  # username

for user_record in tqdm(user_data, desc="Extracting messages"):
    username = user_record.get("username", "unknown")
    user_label = user_record.get("label")
    
    # Skip users without valid labels (0 or 1)
    if user_label not in [0, 1]:
        continue
    
    # Extract posts
    posts = user_record.get("posts", [])
    for post in posts:
        if not isinstance(post, dict):
            continue
        
        title = post.get("title", "").strip()
        content = post.get("content", "").strip()
        
        # Combine title and content
        if is_valid_text(title) and is_valid_text(content):
            message_text = f"{title} {content}"
        elif is_valid_text(title):
            message_text = title
        elif is_valid_text(content):
            message_text = content
        else:
            continue
        
        individual_messages.append(message_text)
        message_labels.append(user_label)
        message_types.append("post")
        message_sources.append(username)
    
    # Extract comments
    comments = user_record.get("comments", [])
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        
        content = comment.get("content", "").strip() or comment.get("body", "").strip()
        
        if is_valid_text(content):
            individual_messages.append(content)
            message_labels.append(user_label)
            message_types.append("comment")
            message_sources.append(username)

print(f"\nExtracted {len(individual_messages):,} individual messages")
print(f"  Posts: {message_types.count('post'):,}")
print(f"  Comments: {message_types.count('comment'):,}")

label_counts = Counter(message_labels)
print(f"\nLabel distribution:")
for label, count in sorted(label_counts.items()):
    label_name = {0: "non-depression", 1: "depression"}.get(label, f"unknown({label})")
    print(f"  {label_name}: {count:,} ({count/len(message_labels)*100:.1f}%)")

# Limit to reasonable size for testing (optional - remove if you want all messages)
max_messages = 10000  # Test on up to 10,000 messages
if len(individual_messages) > max_messages:
    print(f"\nLimiting to {max_messages:,} messages for testing...")
    # Randomly sample while maintaining label distribution
    import random
    random.seed(42)
    indices = list(range(len(individual_messages)))
    random.shuffle(indices)
    indices = indices[:max_messages]
    
    individual_messages = [individual_messages[i] for i in indices]
    message_labels = [message_labels[i] for i in indices]
    message_types = [message_types[i] for i in indices]
    message_sources = [message_sources[i] for i in indices]
    
    print(f"Using {len(individual_messages):,} messages for testing")

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
print("RUNNING PREDICTIONS ON INDIVIDUAL MESSAGES")
print("="*80)
print(f"Processing {len(individual_messages):,} messages...")
print("This may take 30-60 minutes on CPU...\n")

your_predictions = []
your_confidences = []
default_predictions_3class = []
default_confidences_3class = []
default_predictions_binary = []
default_confidences_binary = []

batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move models to device
your_model.to(device)
default_model.to(device)

for i in tqdm(range(0, len(individual_messages), batch_size), desc="Processing batches"):
    batch_texts = individual_messages[i:i+batch_size]
    
    # Your model predictions
    try:
        your_inputs = your_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        your_inputs = {k: v.to(device) for k, v in your_inputs.items()}
        
        with torch.no_grad():
            your_outputs = your_model(**your_inputs)
            your_logits = your_outputs.logits
            your_probs = torch.nn.functional.softmax(your_logits, dim=-1)
            your_preds = torch.argmax(your_logits, dim=-1).cpu().numpy()
            your_conf = your_probs.max(dim=-1)[0].cpu().numpy()
        
        your_predictions.extend(your_preds.tolist())
        your_confidences.extend(your_conf.tolist())
    except Exception as e:
        print(f"Error with your model on batch {i}: {e}")
        your_predictions.extend([-1] * len(batch_texts))
        your_confidences.extend([0.0] * len(batch_texts))
    
    # Default model predictions
    try:
        default_inputs = default_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        default_inputs = {k: v.to(device) for k, v in default_inputs.items()}
        
        with torch.no_grad():
            default_outputs = default_model(**default_inputs)
            default_logits = default_outputs.logits
            default_probs = torch.nn.functional.softmax(default_logits, dim=-1)
            default_preds_3class = torch.argmax(default_logits, dim=-1).cpu().numpy()
            default_conf_3class = default_probs.max(dim=-1)[0].cpu().numpy()
        
        # Map 3-class to binary: 0="not depression"→0, 1="moderate"→1, 2="severe"→1
        default_preds_binary = []
        for pred in default_preds_3class:
            if pred == 0:  # "not depression"
                default_preds_binary.append(0)
            else:  # "moderate" (1) or "severe" (2) → depression
                default_preds_binary.append(1)
        
        default_predictions_3class.extend(default_preds_3class.tolist())
        default_confidences_3class.extend(default_conf_3class.tolist())
        default_predictions_binary.extend(default_preds_binary)
        default_confidences_binary.extend(default_conf_3class.tolist())
    except Exception as e:
        print(f"Error with default model on batch {i}: {e}")
        default_predictions_3class.extend([-1] * len(batch_texts))
        default_confidences_3class.extend([0.0] * len(batch_texts))
        default_predictions_binary.extend([-1] * len(batch_texts))
        default_confidences_binary.extend([0.0] * len(batch_texts))

print("\n" + "="*80)
print("CALCULATING METRICS")
print("="*80)

# Filter out any failed predictions
valid_indices = [i for i in range(len(message_labels)) 
                 if your_predictions[i] != -1 and default_predictions_binary[i] != -1]

if len(valid_indices) < len(message_labels):
    print(f"Warning: {len(message_labels) - len(valid_indices)} predictions failed")
    message_labels = [message_labels[i] for i in valid_indices]
    your_predictions = [your_predictions[i] for i in valid_indices]
    your_confidences = [your_confidences[i] for i in valid_indices]
    default_predictions_3class = [default_predictions_3class[i] for i in valid_indices]
    default_predictions_binary = [default_predictions_binary[i] for i in valid_indices]
    default_confidences_binary = [default_confidences_binary[i] for i in valid_indices]
    individual_messages = [individual_messages[i] for i in valid_indices]
    message_types = [message_types[i] for i in valid_indices]
    message_sources = [message_sources[i] for i in valid_indices]

# Calculate metrics for your model
your_accuracy = accuracy_score(message_labels, your_predictions)
your_f1 = f1_score(message_labels, your_predictions, average='weighted')
your_precision = precision_score(message_labels, your_predictions, average='weighted')
your_recall = recall_score(message_labels, your_predictions, average='weighted')
your_cm = confusion_matrix(message_labels, your_predictions)

# Calculate metrics for default model (binary)
default_accuracy = accuracy_score(message_labels, default_predictions_binary)
default_f1 = f1_score(message_labels, default_predictions_binary, average='weighted')
default_precision = precision_score(message_labels, default_predictions_binary, average='weighted')
default_recall = recall_score(message_labels, default_predictions_binary, average='weighted')
default_cm = confusion_matrix(message_labels, default_predictions_binary)

# Print results
print("\n" + "="*80)
print("RESULTS: INDIVIDUAL MESSAGE CLASSIFICATION")
print("="*80)

print(f"\nTest Set: {len(message_labels):,} individual messages")
print(f"  Posts: {message_types.count('post'):,}")
print(f"  Comments: {message_types.count('comment'):,}")

print("\n" + "-"*80)
print("YOUR TRAINED MODEL")
print("-"*80)
print(f"Accuracy:  {your_accuracy:.4f} ({your_accuracy*100:.2f}%)")
print(f"F1 Score:  {your_f1:.4f} ({your_f1*100:.2f}%)")
print(f"Precision: {your_precision:.4f} ({your_precision*100:.2f}%)")
print(f"Recall:    {your_recall:.4f} ({your_recall*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"                Predicted: 0  Predicted: 1")
print(f"Actual: 0          {your_cm[0][0]:4d}        {your_cm[0][1]:4d}")
print(f"Actual: 1          {your_cm[1][0]:4d}        {your_cm[1][1]:4d}")

print("\n" + "-"*80)
print("DEFAULT DEPROBERTA MODEL (Binary Mapped)")
print("-"*80)
print(f"Accuracy:  {default_accuracy:.4f} ({default_accuracy*100:.2f}%)")
print(f"F1 Score:  {default_f1:.4f} ({default_f1*100:.2f}%)")
print(f"Precision: {default_precision:.4f} ({default_precision*100:.2f}%)")
print(f"Recall:    {default_recall:.4f} ({default_recall*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"                Predicted: 0  Predicted: 1")
print(f"Actual: 0          {default_cm[0][0]:4d}        {default_cm[0][1]:4d}")
print(f"Actual: 1          {default_cm[1][0]:4d}        {default_cm[1][1]:4d}")

print("\n" + "-"*80)
print("COMPARISON")
print("-"*80)
accuracy_diff = your_accuracy - default_accuracy
f1_diff = your_f1 - default_f1
print(f"Accuracy Improvement: {accuracy_diff:+.4f} ({accuracy_diff*100:+.2f}%)")
print(f"F1 Score Improvement:  {f1_diff:+.4f} ({f1_diff*100:+.2f}%)")
if accuracy_diff > 0:
    relative_improvement = (accuracy_diff / default_accuracy) * 100
    print(f"Relative Accuracy Improvement: {relative_improvement:+.2f}%")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save summary
summary = {
    "test_set_size": len(message_labels),
    "posts": message_types.count("post"),
    "comments": message_types.count("comment"),
    "label_distribution": dict(Counter(message_labels)),
    "your_model": {
        "accuracy": float(your_accuracy),
        "f1_score": float(your_f1),
        "precision": float(your_precision),
        "recall": float(your_recall),
        "confusion_matrix": your_cm.tolist()
    },
    "default_model_binary": {
        "accuracy": float(default_accuracy),
        "f1_score": float(default_f1),
        "precision": float(default_precision),
        "recall": float(default_recall),
        "confusion_matrix": default_cm.tolist()
    },
    "comparison": {
        "accuracy_improvement": float(accuracy_diff),
        "accuracy_improvement_percent": float(accuracy_diff * 100),
        "f1_improvement": float(f1_diff),
        "f1_improvement_percent": float(f1_diff * 100),
        "relative_accuracy_improvement": float((accuracy_diff / default_accuracy) * 100) if default_accuracy > 0 else 0
    }
}

summary_file = output_dir / "single_message_comparison_summary.json"
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Saved summary to: {summary_file}")

# Save per-message results
results_file = output_dir / "per_message_predictions.csv"
with open(results_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "message_id", "message_type", "source_user", "true_label",
        "your_prediction", "your_confidence",
        "default_3class", "default_3class_confidence",
        "default_binary", "default_binary_confidence",
        "agreement", "your_correct", "default_correct"
    ])
    
    for i in range(len(message_labels)):
        agreement = "Yes" if your_predictions[i] == default_predictions_binary[i] else "No"
        your_correct = "Yes" if your_predictions[i] == message_labels[i] else "No"
        default_correct = "Yes" if default_predictions_binary[i] == message_labels[i] else "No"
        
        # Truncate message text for CSV
        message_text = individual_messages[i][:200] + "..." if len(individual_messages[i]) > 200 else individual_messages[i]
        
        writer.writerow([
            i, message_types[i], message_sources[i], message_labels[i],
            your_predictions[i], f"{your_confidences[i]:.4f}",
            default_predictions_3class[i], f"{default_confidences_3class[i]:.4f}",
            default_predictions_binary[i], f"{default_confidences_binary[i]:.4f}",
            agreement, your_correct, default_correct
        ])

print(f"Saved per-message results to: {results_file}")
print(f"  Total messages: {len(message_labels):,}")

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80)
print(f"\nResults saved to: {output_dir}")
print(f"  - Summary: {summary_file.name}")
print(f"  - Per-message predictions: {results_file.name}")

