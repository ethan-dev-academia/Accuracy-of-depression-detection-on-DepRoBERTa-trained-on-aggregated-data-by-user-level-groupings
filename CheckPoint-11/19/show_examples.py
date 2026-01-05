"""Quick script to show examples from the labeled dataset."""
import json
from pathlib import Path

output_dir = Path(r"F:\DATA STORAGE\AGG_PACKET\labeling_outputs")

# Load training data
with open(output_dir / "train.json", 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print("="*80)
print("LABELED DATA EXAMPLES")
print("="*80)

# Example 1: Label 0 (Severe/Depression)
print("\n" + "="*80)
print("EXAMPLE 1: Label 0 (Severe/Depression)")
print("="*80)
rec0 = next(r for r in train_data if r['label'] == 0)
print(f"User ID: {rec0['user_id']}")
print(f"Label: {rec0['label']} (severe/depression)")
print(f"Text length: {len(rec0['text']):,} characters")
print(f"\nText preview (first 800 characters):")
print("-" * 80)
print(rec0['text'][:800])
print("...")

# Example 2: Label 1 (Moderate/Anxiety/PTSD)
print("\n" + "="*80)
print("EXAMPLE 2: Label 1 (Moderate/Anxiety/PTSD)")
print("="*80)
rec1 = next(r for r in train_data if r['label'] == 1)
print(f"User ID: {rec1['user_id']}")
print(f"Label: {rec1['label']} (moderate/anxiety/ptsd)")
print(f"Text length: {len(rec1['text']):,} characters")
print(f"\nText preview (first 800 characters):")
print("-" * 80)
print(rec1['text'][:800])
print("...")

# Show statistics
print("\n" + "="*80)
print("DATASET STATISTICS")
print("="*80)
print(f"Total training examples: {len(train_data):,}")
label_counts = {}
for rec in train_data:
    label = rec['label']
    label_counts[label] = label_counts.get(label, 0) + 1

for label, count in sorted(label_counts.items()):
    label_name = {0: "severe (depression)", 1: "moderate (anxiety/ptsd)"}.get(label, f"unknown({label})")
    print(f"  Label {label} ({label_name}): {count:,} examples")

# Show file locations
print("\n" + "="*80)
print("FILE LOCATIONS")
print("="*80)
print(f"Output directory: {output_dir}")
print("\nKey files:")
print(f"  - train.json: {output_dir / 'train.json'}")
print(f"  - val.json: {output_dir / 'val.json'}")
print(f"  - test.json: {output_dir / 'test.json'}")
print(f"  - training_dataset.json: {output_dir / 'training_dataset.json'}")
print(f"  - all_labeled_users.json: {output_dir / 'all_labeled_users.json'}")
print(f"  - validation_report.json: {output_dir / 'validation_report.json'}")

