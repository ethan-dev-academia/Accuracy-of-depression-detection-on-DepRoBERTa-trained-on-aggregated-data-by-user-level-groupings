"""
Build final training set in a new folder with corrected label mapping.

Remaps labels to binary classification:
- 1 = Depression (from depression subreddit)
- 0 = Non-depression (from anxiety/ptsd subreddits)
"""

import json
import shutil
from pathlib import Path
from collections import Counter
import random

# Paths
output_dir = Path(r"F:\DATA STORAGE\AGG_PACKET\labeling_outputs")
final_dir = Path(r"F:\DATA STORAGE\AGG_PACKET\final_training_set")

print("="*80)
print("BUILDING FINAL TRAINING SET")
print("="*80)

# Create new folder
final_dir.mkdir(parents=True, exist_ok=True)
print(f"\nCreated output directory: {final_dir}")

# Load current training data
print("\nLoading current training data...")
train_file = output_dir / "train.json"
val_file = output_dir / "val.json"
test_file = output_dir / "test.json"

if not train_file.exists():
    raise FileNotFoundError(f"Training file not found: {train_file}")

with open(train_file, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open(val_file, 'r', encoding='utf-8') as f:
    val_data = json.load(f)

with open(test_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"Loaded:")
print(f"  Train: {len(train_data):,} examples")
print(f"  Val: {len(val_data):,} examples")
print(f"  Test: {len(test_data):,} examples")

# Show current label distribution
print("\n" + "="*80)
print("CURRENT LABEL DISTRIBUTION")
print("="*80)
train_labels = Counter(r['label'] for r in train_data)
for label, count in sorted(train_labels.items()):
    label_name = {0: "depression (from depression subreddit)", 1: "anxiety/ptsd (from anxiety/ptsd subreddits)"}.get(label, f"unknown({label})")
    print(f"  Label {label} ({label_name}): {count:,}")

# Remap labels: 0->1 (depression), 1->0 (non-depression)
print("\n" + "="*80)
print("REMAPPING LABELS")
print("="*80)
print("""
Remapping strategy:
- Current 0 (depression) -> New 1 (depression)
- Current 1 (anxiety/ptsd) -> New 0 (non-depression)

This creates binary classification:
- 1 = Depression
- 0 = Non-depression
""")

def remap_label(old_label):
    """Remap label: 0->1, 1->0"""
    if old_label == 0:
        return 1  # depression
    elif old_label == 1:
        return 0  # non-depression
    else:
        return old_label  # keep as is (shouldn't happen)

# Remap all datasets
print("\nRemapping labels...")
train_remapped = []
for record in train_data:
    new_record = record.copy()
    new_record['label'] = remap_label(record['label'])
    train_remapped.append(new_record)

val_remapped = []
for record in val_data:
    new_record = record.copy()
    new_record['label'] = remap_label(record['label'])
    val_remapped.append(new_record)

test_remapped = []
for record in test_data:
    new_record = record.copy()
    new_record['label'] = remap_label(record['label'])
    test_remapped.append(new_record)

# Show new label distribution
print("\n" + "="*80)
print("NEW LABEL DISTRIBUTION")
print("="*80)
new_train_labels = Counter(r['label'] for r in train_remapped)
for label, count in sorted(new_train_labels.items()):
    label_name = {0: "non-depression", 1: "depression"}.get(label, f"unknown({label})")
    print(f"  Label {label} ({label_name}): {count:,}")

# Save remapped datasets
print("\n" + "="*80)
print("SAVING FINAL TRAINING SET")
print("="*80)

# Save train
train_output = final_dir / "train.json"
with open(train_output, 'w', encoding='utf-8') as f:
    json.dump(train_remapped, f, indent=2, ensure_ascii=False)
print(f"Saved: {train_output.name} ({len(train_remapped):,} examples)")

# Save val
val_output = final_dir / "val.json"
with open(val_output, 'w', encoding='utf-8') as f:
    json.dump(val_remapped, f, indent=2, ensure_ascii=False)
print(f"Saved: {val_output.name} ({len(val_remapped):,} examples)")

# Save test
test_output = final_dir / "test.json"
with open(test_output, 'w', encoding='utf-8') as f:
    json.dump(test_remapped, f, indent=2, ensure_ascii=False)
print(f"Saved: {test_output.name} ({len(test_remapped):,} examples)")

# Create full dataset
full_remapped = train_remapped + val_remapped + test_remapped
full_output = final_dir / "training_dataset.json"
with open(full_output, 'w', encoding='utf-8') as f:
    json.dump(full_remapped, f, indent=2, ensure_ascii=False)
print(f"Saved: {full_output.name} ({len(full_remapped):,} examples)")

# Save statistics
stats = {
    "total_examples": len(full_remapped),
    "label_distribution": dict(Counter(r['label'] for r in full_remapped)),
    "label_mapping": {
        "0": "non-depression",
        "1": "depression"
    },
    "remapping_applied": True,
    "original_mapping": {
        "0": "depression (from depression subreddit)",
        "1": "anxiety/ptsd (from anxiety/ptsd subreddits)"
    },
    "split_sizes": {
        "train": len(train_remapped),
        "val": len(val_remapped),
        "test": len(test_remapped)
    },
    "average_text_length": sum(len(r['text']) for r in full_remapped) / len(full_remapped) if full_remapped else 0
}

stats_output = final_dir / "dataset_stats.json"
with open(stats_output, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(f"Saved: {stats_output.name}")

# Create HuggingFace format if available
try:
    from datasets import Dataset
    
    print("\nCreating HuggingFace Dataset format...")
    
    # Train
    ds_train = Dataset.from_list(train_remapped)
    ds_train.save_to_disk(str(final_dir / "train_hf"))
    print(f"Saved: train_hf/ ({len(ds_train):,} examples)")
    
    # Val
    ds_val = Dataset.from_list(val_remapped)
    ds_val.save_to_disk(str(final_dir / "val_hf"))
    print(f"Saved: val_hf/ ({len(ds_val):,} examples)")
    
    # Test
    ds_test = Dataset.from_list(test_remapped)
    ds_test.save_to_disk(str(final_dir / "test_hf"))
    print(f"Saved: test_hf/ ({len(ds_test):,} examples)")
    
    # Full
    ds_full = Dataset.from_list(full_remapped)
    ds_full.save_to_disk(str(final_dir / "training_dataset_hf"))
    print(f"Saved: training_dataset_hf/ ({len(ds_full):,} examples)")
    
except ImportError:
    print("\nHuggingFace datasets not available, skipping HF format")

# Create label mapping file
label_mapping = {
    "label_mapping": {
        "0": "non-depression",
        "1": "depression"
    },
    "description": "Binary classification labels",
    "remapped_from": {
        "original_0": "depression (from depression subreddit) -> new 1",
        "original_1": "anxiety/ptsd (from anxiety/ptsd subreddits) -> new 0"
    }
}

mapping_output = final_dir / "label_mapping.json"
with open(mapping_output, 'w', encoding='utf-8') as f:
    json.dump(label_mapping, f, indent=2, ensure_ascii=False)
print(f"Saved: {mapping_output.name}")

# Summary
print("\n" + "="*80)
print("FINAL TRAINING SET SUMMARY")
print("="*80)
print(f"\nLocation: {final_dir}")
print(f"\nFiles created:")
print(f"  - train.json: {len(train_remapped):,} examples")
print(f"  - val.json: {len(val_remapped):,} examples")
print(f"  - test.json: {len(test_remapped):,} examples")
print(f"  - training_dataset.json: {len(full_remapped):,} examples")
print(f"  - dataset_stats.json: Statistics")
print(f"  - label_mapping.json: Label mapping info")

print(f"\nLabel Distribution (Final):")
for label, count in sorted(new_train_labels.items()):
    label_name = {0: "non-depression", 1: "depression"}.get(label, f"unknown({label})")
    percentage = (count / len(train_remapped)) * 100
    print(f"  Label {label} ({label_name}): {count:,} ({percentage:.1f}%)")

print("\n" + "="*80)
print("SUCCESS: Final training set created!")
print("="*80)
print(f"\nYour final training set is ready in: {final_dir}")
print("\nLabel mapping:")
print("  1 = Depression")
print("  0 = Non-depression")




