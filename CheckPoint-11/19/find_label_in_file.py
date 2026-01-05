"""Show exactly where the label field is in the JSON file."""
import json
from pathlib import Path

output_dir = Path(r"F:\DATA STORAGE\AGG_PACKET\labeling_outputs")

print("="*80)
print("FINDING THE LABEL FIELD IN all_labeled_users.json")
print("="*80)

with open(output_dir / "all_labeled_users.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get first record
rec = data[0]
print(f"\nUser: {rec.get('username')}")
print(f"\nRecord structure (in order):")
print("-" * 80)

# Show the order of fields
keys = list(rec.keys())
for i, key in enumerate(keys, 1):
    value = rec[key]
    if key == 'posts':
        print(f"{i}. '{key}': [array with {len(value)} posts]")
    elif key == 'comments':
        print(f"{i}. '{key}': [array with {len(value)} comments]")
    elif key == 'label':
        print(f"{i}. '{key}': {value}  <-- HERE IS THE LABEL!")
    else:
        print(f"{i}. '{key}': {value}")

print("\n" + "="*80)
print("IMPORTANT: The label is at the END of each record")
print("="*80)
print("""
The label field appears AFTER all the posts and comments arrays.

In the JSON file, the structure looks like:
{
  "username": "...",
  "exists": true,
  "total_posts": 4,
  "total_comments": 96,
  "posts": [
    { ... post 1 ... },
    { ... post 2 ... },
    { ... post 3 ... },
    { ... post 4 ... }
  ],
  "comments": [
    { ... comment 1 ... },
    { ... comment 2 ... },
    ... (94 more comments) ...
    { ... comment 96 ... }
  ],
  "last_activity": 1754995803.0,
  "label": 0  <-- THE LABEL IS HERE, AT THE END!
}
""")

print("\n" + "="*80)
print("HOW TO FIND IT IN YOUR EDITOR:")
print("="*80)
print("""
1. Scroll down to the END of each user object
2. Look for the closing bracket ] of the "comments" array
3. After that, you'll see:
   "last_activity": ...,
   "label": 0  <-- This is it!
4. The label is the LAST field before the closing brace }
""")

# Show a sample with label
print("\n" + "="*80)
print("SAMPLE RECORD SHOWING LABEL:")
print("="*80)
sample = {
    "username": rec.get('username'),
    "total_posts": rec.get('total_posts'),
    "total_comments": rec.get('total_comments'),
    "last_activity": rec.get('last_activity'),
    "label": rec.get('label')  # <-- The label
}
print(json.dumps(sample, indent=2, ensure_ascii=False))

# Check if label exists in all records
print("\n" + "="*80)
print("CHECKING ALL RECORDS:")
print("="*80)
records_with_label = sum(1 for r in data if 'label' in r and r['label'] is not None)
records_without_label = sum(1 for r in data if 'label' not in r or r['label'] is None)

print(f"Total records: {len(data)}")
print(f"Records WITH label: {records_with_label}")
print(f"Records WITHOUT label: {records_without_label}")

if records_without_label > 0:
    print(f"\nWARNING: {records_without_label} records don't have labels!")
    print("These are users that couldn't be matched to the RMH dataset.")

