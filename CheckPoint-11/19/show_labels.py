"""Show where labels are stored in the data."""
import json
from pathlib import Path
from collections import Counter

output_dir = Path(r"F:\DATA STORAGE\AGG_PACKET\labeling_outputs")

print("="*80)
print("WHERE ARE THE LABELS STORED?")
print("="*80)

# Check training data
print("\n1. TRAINING DATA (train.json, val.json, test.json):")
print("-" * 80)
with open(output_dir / "train.json", 'r', encoding='utf-8') as f:
    train_data = json.load(f)

sample = train_data[0]
print(f"Record structure: {list(sample.keys())}")
print(f"\nSample record:")
print(f"  'label': {sample['label']}  <-- THIS IS THE LABEL")
print(f"  'user_id': {sample['user_id']}")
print(f"  'text': '{sample['text'][:100]}...'")
print(f"  'segments': {len(sample['segments'])} segments")

# Label distribution
labels = [r['label'] for r in train_data]
counts = Counter(labels)
print(f"\nLabel distribution in train.json:")
for label, count in sorted(counts.items()):
    label_name = {0: "severe (depression)", 1: "moderate (anxiety/ptsd)"}.get(label, f"unknown({label})")
    print(f"  Label {label} ({label_name}): {count:,} examples")

# Check raw data
print("\n" + "="*80)
print("2. RAW DATA (all_labeled_users.json):")
print("-" * 80)
with open(output_dir / "all_labeled_users.json", 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

sample_raw = raw_data[0]
print(f"Record structure: {list(sample_raw.keys())}")
print(f"\nSample record:")
print(f"  'username': {sample_raw.get('username')}")
print(f"  'label': {sample_raw.get('label')}  <-- THIS IS THE LABEL")
print(f"  'posts': {len(sample_raw.get('posts', []))} posts")
print(f"  'comments': {len(sample_raw.get('comments', []))} comments")

# Label distribution in raw data
raw_labels = [r.get('label') for r in raw_data if r.get('label') is not None]
raw_counts = Counter(raw_labels)
print(f"\nLabel distribution in all_labeled_users.json:")
for label, count in sorted(raw_counts.items()):
    label_name = {0: "severe (depression)", 1: "moderate (anxiety/ptsd)"}.get(label, f"unknown({label})")
    print(f"  Label {label} ({label_name}): {count:,} examples")

# Show examples with different labels
print("\n" + "="*80)
print("3. LABEL EXAMPLES:")
print("-" * 80)

# Find examples
label0_example = next((r for r in raw_data if r.get('label') == 0), None)
label1_example = next((r for r in raw_data if r.get('label') == 1), None)

if label0_example:
    print(f"\nLabel 0 (Severe/Depression) Example:")
    print(f"  Username: {label0_example.get('username')}")
    print(f"  Label: {label0_example.get('label')}")
    print(f"  Posts: {len(label0_example.get('posts', []))}")
    print(f"  Comments: {len(label0_example.get('comments', []))}")

if label1_example:
    print(f"\nLabel 1 (Moderate/Anxiety/PTSD) Example:")
    print(f"  Username: {label1_example.get('username')}")
    print(f"  Label: {label1_example.get('label')}")
    print(f"  Posts: {len(label1_example.get('posts', []))}")
    print(f"  Comments: {len(label1_example.get('comments', []))}")

# Show label mapping
print("\n" + "="*80)
print("4. LABEL MAPPING (What each number means):")
print("-" * 80)
print("""
Label 0 = Severe Depression
  - Users from 'depression' subreddit
  - Indicates severe mental health condition

Label 1 = Moderate Anxiety/PTSD
  - Users from 'anxiety', 'ptsd', 'healthanxiety', 'socialanxiety' subreddits
  - Indicates moderate mental health condition

Label 2 = Not Depression (if present)
  - Users from other subreddits (adhd, addiction, etc.)
  - Control group or other conditions
  - Note: Currently only labels 0 and 1 are in your training data
""")

# Show where labels come from
print("="*80)
print("5. WHERE LABELS CAME FROM:")
print("-" * 80)
print("""
Labels were extracted from RMH (Reddit Mental Health) dataset:
1. RMH CSV files contain usernames and subreddit information
2. Subreddit name determines the label:
   - 'depression' subreddit -> Label 0
   - 'anxiety', 'ptsd' subreddits -> Label 1
3. Labels were matched to your AGG_PACKET users by username
4. Added as 'label' field in each user record
""")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
LABEL LOCATION:
- Field name: 'label'
- Data type: Integer (0, 1, or 2)
- Location: Top-level field in each record

IN TRAINING DATA:
- train.json, val.json, test.json: Each record has 'label' field
- Example: {"text": "...", "label": 0, "user_id": "...", "segments": [...]}

IN RAW DATA:
- all_labeled_users.json: Each user record has 'label' field
- Example: {"username": "...", "label": 0, "posts": [...], "comments": [...]}

LABEL VALUES:
- 0 = Severe Depression
- 1 = Moderate Anxiety/PTSD
- 2 = Not Depression (if present)
""")

