"""
Utility Script: Inspect labeled data quickly.

This script provides a quick way to explore labeled data.
"""

import json
from pathlib import Path
from collections import Counter

def load_config(config_path="labeling_config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def inspect_labeled_data(agg_packet_dir, output_dir, num_samples=5):
    """Inspect labeled data."""
    agg_path = Path(agg_packet_dir)
    output_path = Path(output_dir)
    
    # Try to find labeled files
    json_files = sorted(agg_path.glob("reddit_user_analysis_*_labeled.json"))
    consolidated_file = output_path / "all_labeled_users.json"
    
    if consolidated_file.exists():
        print(f"Loading from consolidated file: {consolidated_file.name}")
        with open(consolidated_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
    elif json_files:
        print(f"Loading from {len(json_files)} labeled file(s)...")
        records = []
        for json_file in json_files[:1]:  # Just first file for inspection
            with open(json_file, 'r', encoding='utf-8') as f:
                file_records = json.load(f)
                records.extend([r for r in file_records if r.get("label") is not None])
    else:
        print("No labeled data found!")
        return
    
    print(f"\nTotal labeled records: {len(records):,}")
    
    # Label distribution
    label_counts = Counter(r.get("label") for r in records)
    print(f"\nLabel Distribution:")
    for label, count in sorted(label_counts.items()):
        label_name = {0: "severe (depression)", 1: "moderate (anxiety/ptsd)", 2: "not depression"}.get(label, f"unknown({label})")
        print(f"  {label_name} ({label}): {count:,}")
    
    # Sample records
    print(f"\n{'='*60}")
    print(f"Sample Records (showing {num_samples} per label):")
    print(f"{'='*60}")
    
    by_label = {}
    for record in records:
        label = record.get("label")
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(record)
    
    for label in sorted(by_label.keys()):
        label_name = {0: "severe (depression)", 1: "moderate (anxiety/ptsd)", 2: "not depression"}.get(label, f"unknown({label})")
        print(f"\nLabel {label} ({label_name}):")
        print("-" * 60)
        
        samples = by_label[label][:num_samples]
        for idx, record in enumerate(samples, 1):
            username = record.get("username", "unknown")
            posts = record.get("posts", [])
            comments = record.get("comments", [])
            
            print(f"\n  Sample {idx}: {username}")
            print(f"    Posts: {len(posts)}")
            print(f"    Comments: {len(comments)}")
            
            # Show first post/comment
            if posts:
                first_post = posts[0]
                title = first_post.get("title", "")[:100]
                content = first_post.get("content", "")[:100]
                print(f"    First post title: {title}...")
                if content:
                    print(f"    First post content: {content}...")
            elif comments:
                first_comment = comments[0]
                content = (first_comment.get("content") or first_comment.get("body", ""))[:200]
                print(f"    First comment: {content}...")

def main():
    """Main execution function."""
    print("="*60)
    print("Inspect Labeled Data")
    print("="*60)
    
    try:
        config = load_config()
        agg_packet_dir = config['paths']['agg_packet']
        output_dir = config['paths']['output_dir']
        
        inspect_labeled_data(agg_packet_dir, output_dir)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

