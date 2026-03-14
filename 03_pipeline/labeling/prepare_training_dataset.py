"""
Script 4: Prepare training dataset from labeled data.

This script aggregates user content and creates training-ready dataset formats.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import random
import sys

try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not found. HuggingFace Dataset format will be skipped.")

def load_config(config_path="labeling_config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_valid_text(text):
    """Check if text is valid."""
    if not text or not isinstance(text, str):
        return False
    text_lower = text.strip().lower()
    return text_lower not in ["[removed]", "[deleted]", ""]

def aggregate_user_content(record):
    """
    Aggregate posts and comments into single text representation.
    
    Returns:
        tuple: (aggregated_text, segments_list)
    """
    posts = record.get("posts", [])
    comments = record.get("comments", [])
    
    text_parts = []
    segments = []
    
    # Process posts
    for post in posts:
        if not isinstance(post, dict):
            continue
        
        title = post.get("title", "").strip()
        content = post.get("content", "").strip()
        
        if is_valid_text(title) and is_valid_text(content):
            combined = f"{title} {content}"
        elif is_valid_text(title):
            combined = title
        elif is_valid_text(content):
            combined = content
        else:
            continue
        
        text_parts.append(combined)
        segments.append(combined[:200])  # First 200 chars for segments
    
    # Process comments
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        
        content = comment.get("content", "").strip() or comment.get("body", "").strip()
        
        if is_valid_text(content):
            text_parts.append(content)
            segments.append(content[:200])
    
    # Combine all text
    aggregated_text = " ".join(text_parts)
    aggregated_text = " ".join(aggregated_text.split())  # Normalize whitespace
    
    return aggregated_text, segments

def prepare_training_dataset(agg_packet_dir, output_dir, training_config):
    """
    Prepare training dataset from labeled data.
    
    Args:
        agg_packet_dir: Path to AGG_PACKET directory
        output_dir: Directory to save outputs
        training_config: Training preparation configuration
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Try to load filtered dataset first, otherwise load from labeled files
    filtered_file = output_path / "filtered_labeled_data.json"
    consolidated_file = output_path / "all_labeled_users.json"
    
    if filtered_file.exists():
        print(f"Loading filtered dataset from: {filtered_file.name}")
        with open(filtered_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
    elif consolidated_file.exists():
        print(f"Loading consolidated dataset from: {consolidated_file.name}")
        with open(consolidated_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
    else:
        # Load from labeled JSON files
        agg_path = Path(agg_packet_dir)
        json_files = sorted(agg_path.glob("reddit_user_analysis_*_labeled.json"))
        
        if not json_files:
            raise FileNotFoundError("No labeled data found. Run previous scripts first!")
        
        print(f"Loading from {len(json_files)} labeled file(s)...")
        records = []
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_records = json.load(f)
                # Filter to only users with valid labels
                records.extend([r for r in file_records if r.get("label") is not None and r.get("label") in [0, 1, 2]])
    
    print(f"Loaded {len(records):,} labeled user records")
    print("="*60)
    
    # Aggregate content
    print("\nAggregating user content...")
    dataset_records = []
    empty_users = 0
    
    for record in records:
        username = record.get("username", "unknown")
        label = record.get("label")
        
        if label not in [0, 1, 2]:
            continue
        
        aggregated_text, segments = aggregate_user_content(record)
        
        if not aggregated_text or len(aggregated_text.strip()) < 10:
            empty_users += 1
            continue
        
        dataset_records.append({
            "text": aggregated_text,
            "label": label,
            "user_id": username,
            "segments": segments
        })
    
    print(f"Aggregated {len(dataset_records):,} users with valid content")
    print(f"Skipped {empty_users} users with insufficient content")
    
    # Calculate statistics
    label_counts = Counter(r["label"] for r in dataset_records)
    avg_length = sum(len(r["text"]) for r in dataset_records) / len(dataset_records) if dataset_records else 0
    
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {len(dataset_records):,}")
    print(f"  Average text length: {avg_length:.0f} characters")
    print(f"  Label distribution:")
    for label, count in sorted(label_counts.items()):
        label_name = {0: "severe (depression)", 1: "moderate (anxiety/ptsd)", 2: "not depression"}.get(label, f"unknown({label})")
        print(f"    {label_name} ({label}): {count:,}")
    
    # Split dataset
    train_ratio = training_config['train_ratio']
    val_ratio = training_config['val_ratio']
    test_ratio = training_config['test_ratio']
    random_seed = training_config['random_seed']
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        print(f"\nWarning: Split ratios don't sum to 1.0, normalizing...")
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    
    print(f"\nSplitting dataset:")
    print(f"  Train: {train_ratio*100:.1f}%")
    print(f"  Val: {val_ratio*100:.1f}%")
    print(f"  Test: {test_ratio*100:.1f}%")
    
    # Stratified split by label
    random.seed(random_seed)
    
    # Group by label
    by_label = defaultdict(list)
    for record in dataset_records:
        by_label[record["label"]].append(record)
    
    train_records = []
    val_records = []
    test_records = []
    
    for label, label_records in by_label.items():
        random.shuffle(label_records)
        
        n = len(label_records)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_records.extend(label_records[:n_train])
        val_records.extend(label_records[n_train:n_train+n_val])
        test_records.extend(label_records[n_train+n_val:])
    
    # Shuffle splits
    random.shuffle(train_records)
    random.shuffle(val_records)
    random.shuffle(test_records)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_records):,}")
    print(f"  Val: {len(val_records):,}")
    print(f"  Test: {len(test_records):,}")
    
    # Save outputs
    output_formats = training_config.get('output_formats', ['json'])
    
    if 'json' in output_formats:
        # Save full dataset
        full_file = output_path / "training_dataset.json"
        with open(full_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_records, f, indent=2, ensure_ascii=False)
        print(f"\nSaved full dataset to: {full_file}")
        
        # Save splits
        train_file = output_path / "train.json"
        val_file = output_path / "val.json"
        test_file = output_path / "test.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_records, f, indent=2, ensure_ascii=False)
        print(f"Saved train split to: {train_file}")
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_records, f, indent=2, ensure_ascii=False)
        print(f"Saved validation split to: {val_file}")
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_records, f, indent=2, ensure_ascii=False)
        print(f"Saved test split to: {test_file}")
    
    if 'huggingface' in output_formats and HAS_DATASETS:
        # Create HuggingFace Dataset
        print(f"\nCreating HuggingFace Dataset format...")
        
        # Full dataset
        ds_full = Dataset.from_list(dataset_records)
        ds_full.save_to_disk(str(output_path / "training_dataset_hf"))
        print(f"Saved HuggingFace dataset to: training_dataset_hf/")
        
        # Splits
        ds_train = Dataset.from_list(train_records)
        ds_val = Dataset.from_list(val_records)
        ds_test = Dataset.from_list(test_records)
        
        ds_train.save_to_disk(str(output_path / "train_hf"))
        ds_val.save_to_disk(str(output_path / "val_hf"))
        ds_test.save_to_disk(str(output_path / "test_hf"))
        
        print(f"Saved HuggingFace splits to: train_hf/, val_hf/, test_hf/")
    
    # Save statistics
    stats = {
        "total_examples": len(dataset_records),
        "label_distribution": dict(label_counts),
        "average_text_length": round(avg_length, 2),
        "split_sizes": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records)
        },
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio
        },
        "random_seed": random_seed
    }
    
    stats_file = output_path / "training_dataset_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nSaved statistics to: {stats_file}")
    
    return stats

def main():
    """Main execution function."""
    print("="*60)
    print("Script 4: Prepare Training Dataset")
    print("="*60)
    
    try:
        config = load_config(str(Path(__file__).parent / "labeling_config.json"))
        agg_packet_dir = config['paths']['agg_packet']
        output_dir = config['paths']['output_dir']
        training_config = config['training_prep']
        
        stats = prepare_training_dataset(agg_packet_dir, output_dir, training_config)
        
        print(f"\n{'='*60}")
        print("SUCCESS: Training dataset prepared!")
        print(f"{'='*60}")
        print("Your data is now ready for model training!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

