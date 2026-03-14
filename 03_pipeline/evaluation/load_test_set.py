"""
Quick script to load and inspect the test set.
"""
import json
from pathlib import Path
from collections import Counter

# Test set location
test_file = Path(r"F:\DATA STORAGE\AGG_PACKET\final_training_set\test.json")

print("="*80)
print("LOADING TEST SET")
print("="*80)
print(f"\nFile: {test_file}")
print(f"Exists: {test_file.exists()}")

if not test_file.exists():
    print("\n[ERROR] Test file not found!")
    exit(1)

# Load test set
print("\nLoading test set...")
with open(test_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data):,} examples")

# Show label distribution
labels = Counter(record['label'] for record in test_data)
print("\nLabel distribution:")
for label, count in sorted(labels.items()):
    label_name = {0: "non-depression", 1: "depression"}.get(label, f"unknown({label})")
    percentage = (count / len(test_data)) * 100
    print(f"  Label {label} ({label_name}): {count:,} ({percentage:.1f}%)")

# Show example
print("\n" + "="*80)
print("EXAMPLE FROM TEST SET")
print("="*80)
if test_data:
    example = test_data[0]
    label_name = {0: "non-depression", 1: "depression"}.get(example['label'], "unknown")
    print(f"\nLabel: {example['label']} ({label_name})")
    print(f"Text length: {len(example['text'])} characters")
    print(f"\nText preview (first 200 chars):")
    print(f"  {example['text'][:200]}...")

print("\n" + "="*80)
print("TEST SET READY FOR EVALUATION")
print("="*80)




