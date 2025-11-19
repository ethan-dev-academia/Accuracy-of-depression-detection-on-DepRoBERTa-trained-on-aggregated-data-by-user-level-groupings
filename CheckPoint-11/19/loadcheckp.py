import os
import json
from pathlib import Path
from datasets import Dataset
from collections import Counter
from util import inspect_dataset

# Path to your dataset directory
DATASET_DIR = r"F:\DATA STORAGE\AGG_PACKET"

# You can specify a specific file or let it auto-detect
# Leave as None to use the first JSON file found, or specify a filename
SPECIFIC_FILE = None  # e.g., "reddit_user_analysis_20250813_174106.json"

def load_user_records(dataset_dir, specific_file=None, limit=None):
    """Load Reddit user records from JSON files in the directory."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Find JSON files
    if specific_file:
        json_files = list(dataset_path.glob(specific_file))
        if not json_files:
            raise FileNotFoundError(f"Specific file not found: {specific_file}")
    else:
        json_files = sorted(dataset_path.glob("reddit_user_analysis_*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {dataset_dir}")
    
    print(f"Found {len(json_files)} JSON file(s). Loading from: {json_files[0].name}")
    
    # Load the first (or specified) file
    json_file = json_files[0]
    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    if not isinstance(records, list):
        raise ValueError(f"Expected JSON file to contain a list, got {type(records)}")
    
    # Apply limit if specified
    if limit and limit > 0:
        records = records[:limit]
        print(f"Limited to first {limit} records")
    
    print(f"Loaded {len(records)} user records from {json_file.name}")
    return records

def aggregate_user_posts(records):
    """Convert Reddit user records to aggregated text format for the model."""
    aggregated_texts = []
    user_ids = []
    segments_list = []
    
    for record in records:
        username = record.get("username", "unknown")
        
        # Skip users with errors
        if record.get("error"):
            continue
        
        # Aggregate posts and comments
        text_parts = []
        segments = []
        
        # Collect posts
        posts = record.get("posts") or []
        for post in posts:
            content = post.get("content", "").strip()
            title = post.get("title", "").strip()
            
            # Skip removed/deleted posts
            if not content or content.lower() in ["[removed]", "[deleted]", ""]:
                if title:
                    text_parts.append(title)
                    segments.append(title)
                continue
            
            # Combine title and content if both exist
            if title and content:
                combined = f"{title} {content}"
            else:
                combined = content or title
            
            text_parts.append(combined)
            segments.append(combined[:200])  # Keep first 200 chars for segments
        
        # Collect comments
        comments = record.get("comments") or []
        for comment in comments:
            content = comment.get("content", "").strip() or comment.get("body", "").strip()
            
            # Skip removed/deleted comments
            if not content or content.lower() in ["[removed]", "[deleted]", ""]:
                continue
            
            text_parts.append(content)
            segments.append(content[:200])  # Keep first 200 chars for segments
        
        # Skip users with no valid content
        if not text_parts:
            continue
        
        # Combine all text parts
        combined_text = " ".join(text_parts)
        combined_text = " ".join(combined_text.split())  # Normalize whitespace
        
        aggregated_texts.append(combined_text)
        user_ids.append(username)
        segments_list.append(segments)
    
    print(f"Aggregated {len(aggregated_texts)} users with valid content")
    
    # Create HuggingFace Dataset
    ds = Dataset.from_dict({
        "text": aggregated_texts,
        "user_id": user_ids,
        "segments": segments_list,
    })
    
    return ds

# Load and process the dataset
print("=" * 60)
print("Loading dataset from:", DATASET_DIR)
print("=" * 60)

try:
    # Load raw records
    records = load_user_records(DATASET_DIR, specific_file=SPECIFIC_FILE, limit=500)
    
    # Aggregate into dataset format
    ds = aggregate_user_posts(records)
    
    print(f"\nSuccessfully created dataset with {len(ds)} examples")
    print(f"Dataset columns: {ds.column_names}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Now inspect
print("\n" + "=" * 60)
print("Dataset Inspection")
print("=" * 60)
inspect_dataset(ds)

# Check for label column (your datasets don't have labels currently)
print("\n" + "=" * 60)
print("Label Column Check")
print("=" * 60)
possible_label_cols = ['label', 'labels', 'target', 'truth', 'y', 'annotation', 'gold_label']
found = [c for c in ds.column_names if c in possible_label_cols]

if found:
    label_col = found[0]
    print(f"Found label column: {label_col}")
    # If labels are text (e.g., 'not depression','moderate','severe'), map to integers:
    sample_vals = ds[label_col][:100]
    # detect if text labels
    if any(isinstance(v, str) for v in sample_vals):
        unique_vals = list(dict.fromkeys([v for v in sample_vals if v is not None]))
        print("Sample text labels (first 20 unique):", unique_vals[:20])
        # provide mapping automatically (alphabetic order to reproducible ints)
        mapping = {v: i for i, v in enumerate(sorted(unique_vals))}
        print("Auto mapping (text->int):", mapping)
        ds = ds.map(lambda x: {'label': mapping.get(x[label_col], -1)})
    else:
        # numeric labels: coerce to int and copy
        ds = ds.map(lambda x: {'label': int(x[label_col]) if x[label_col] is not None else -1})
    
    # show label distribution (sample)
    lbls = ds['label'][:5000] if len(ds) > 5000 else ds['label']
    print("Label counts (sample):", Counter(lbls))
else:
    # No label column found - this is expected for your current dataset
    print("No standard label column found in the dataset.")
    print("This is expected - your Reddit user JSON files don't contain labels.")
    print("Creating placeholder 'label' = -1 for all rows for compatibility.")
    ds = ds.map(lambda x: {'label': -1})

print("\n" + "=" * 60)
print("Dataset Summary")
print("=" * 60)
print(f"Total examples: {len(ds)}")
print(f"Columns: {ds.column_names}")
if 'label' in ds.column_names:
    lbls = ds['label'][:min(5000, len(ds))] if len(ds) > 5000 else ds['label']
    print(f"Label distribution: {Counter(lbls)}")

print("\nDataset is ready to use!")
print("You can now use 'ds' variable for further processing.")
