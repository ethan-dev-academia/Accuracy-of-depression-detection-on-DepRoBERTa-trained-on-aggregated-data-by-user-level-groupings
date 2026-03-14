"""Verify current label mapping and check what the model expects."""
import json
from pathlib import Path

print("="*80)
print("VERIFYING LABEL MAPPING")
print("="*80)

# Check model config
model_config_path = Path("ModelB_final/config.json")
if model_config_path.exists():
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    print("\nPretrained Model Label Mapping:")
    print(f"  id2label: {model_config.get('id2label', {})}")
    print(f"  label2id: {model_config.get('label2id', {})}")
    
    # Model expects:
    # 0 = "severe" (depression)
    # 1 = "moderate" (anxiety/ptsd)
    # 2 = "not depression"
    
    print("\nModel expects:")
    print("  0 = severe (depression)")
    print("  1 = moderate (anxiety/ptsd)")
    print("  2 = not depression")
else:
    print("\nModel config not found - checking training data labels...")

# Check current training data
output_dir = Path(r"F:\DATA STORAGE\AGG_PACKET\labeling_outputs")
train_file = output_dir / "train.json"

if train_file.exists():
    print("\n" + "="*80)
    print("CURRENT TRAINING DATA LABELS")
    print("="*80)
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Sample records
    rec0 = next((r for r in train_data if r['label'] == 0), None)
    rec1 = next((r for r in train_data if r['label'] == 1), None)
    
    if rec0:
        print(f"\nLabel 0 example:")
        print(f"  User: {rec0['user_id']}")
        print(f"  Text preview: {rec0['text'][:150]}...")
        print(f"  Source: From 'depression' subreddit")
    
    if rec1:
        print(f"\nLabel 1 example:")
        print(f"  User: {rec1['user_id']}")
        print(f"  Text preview: {rec1['text'][:150]}...")
        print(f"  Source: From 'anxiety/ptsd' subreddits")
    
    # Count labels
    from collections import Counter
    labels = [r['label'] for r in train_data]
    counts = Counter(labels)
    print(f"\nCurrent label distribution:")
    for label, count in sorted(counts.items()):
        print(f"  Label {label}: {count:,} examples")

print("\n" + "="*80)
print("LABEL MAPPING ANALYSIS")
print("="*80)
print("""
CURRENT MAPPING (from RMH dataset):
- Label 0 = Users from 'depression' subreddit (SEVERE DEPRESSION)
- Label 1 = Users from 'anxiety/ptsd' subreddits (MODERATE/ANXIETY)

USER REQUEST:
- Label 1 = Depression
- Label 0 = Non-depression

DECISION NEEDED:
Since you want binary classification (depression vs non-depression):
- Current 0 (depression) -> Should become 1 (depression)
- Current 1 (anxiety/ptsd) -> Should become 0 (non-depression)

This makes sense because:
- Depression = 1 (positive class)
- Non-depression (including anxiety/ptsd) = 0 (negative class)
""")




