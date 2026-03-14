"""
Script 3: Validate labeled dataset for training readiness.

This script checks data quality, label distribution, and content quality.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import sys

def load_config(config_path="labeling_config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_valid_text(text):
    """Check if text is valid (not removed/deleted/empty)."""
    if not text or not isinstance(text, str):
        return False
    text_lower = text.strip().lower()
    return text_lower not in ["[removed]", "[deleted]", ""]

def validate_labeled_data(agg_packet_dir, output_dir, validation_config):
    """
    Validate labeled dataset.
    
    Args:
        agg_packet_dir: Path to AGG_PACKET directory
        output_dir: Directory to save outputs
        validation_config: Validation configuration dict
    """
    agg_path = Path(agg_packet_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find labeled JSON files
    json_files = sorted(agg_path.glob("reddit_user_analysis_*_labeled.json"))
    
    if not json_files:
        # Try consolidated file
        consolidated_file = output_path / "all_labeled_users.json"
        if consolidated_file.exists():
            json_files = [consolidated_file]
        else:
            raise FileNotFoundError("No labeled JSON files found. Run 'match_labels_to_agg_packet.py' first!")
    
    print(f"Found {len(json_files)} labeled file(s)")
    print("="*60)
    
    min_samples = validation_config['min_samples_per_class']
    min_content_length = validation_config['min_content_length']
    max_imbalance = validation_config['max_class_imbalance_ratio']
    
    # Statistics
    total_users = 0
    valid_labeled_users = 0
    users_with_valid_content = 0
    label_counts = Counter()
    content_lengths = []
    issues = defaultdict(list)
    
    all_valid_users = []
    
    # Process each file
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except Exception as e:
            print(f"  [ERROR] Error reading file: {e}")
            continue
        
        if not isinstance(records, list):
            print(f"  Warning: Expected list, got {type(records)}")
            continue
        
        print(f"  Analyzing {len(records):,} records...")
        
        for record in records:
            total_users += 1
            
            # Check required fields
            username = record.get("username")
            label = record.get("label")
            posts = record.get("posts", [])
            comments = record.get("comments", [])
            
            # Check label validity
            if label is None:
                issues["missing_label"].append(username)
                continue
            
            if label not in [0, 1, 2]:
                issues["invalid_label"].append((username, label))
                continue
            
            valid_labeled_users += 1
            label_counts[label] += 1
            
            # Check content quality
            text_parts = []
            
            for post in posts:
                if isinstance(post, dict):
                    title = post.get("title", "")
                    content = post.get("content", "")
                    if is_valid_text(title):
                        text_parts.append(title)
                    if is_valid_text(content):
                        text_parts.append(content)
            
            for comment in comments:
                if isinstance(comment, dict):
                    content = comment.get("content") or comment.get("body", "")
                    if is_valid_text(content):
                        text_parts.append(content)
            
            combined_text = " ".join(text_parts)
            text_length = len(combined_text.strip())
            content_lengths.append(text_length)
            
            if text_length >= min_content_length:
                users_with_valid_content += 1
                all_valid_users.append(record)
            else:
                issues["insufficient_content"].append((username, text_length))
    
    # Calculate statistics
    print(f"\n{'='*60}")
    print("Validation Results")
    print(f"{'='*60}")
    
    print(f"\nBasic Statistics:")
    print(f"  Total users: {total_users:,}")
    print(f"  Valid labeled users: {valid_labeled_users:,}")
    print(f"  Users with sufficient content: {users_with_valid_content:,}")
    
    print(f"\nLabel Distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        label_name = {0: "severe (depression)", 1: "moderate (anxiety/ptsd)", 2: "not depression"}.get(label, f"unknown({label})")
        status = "OK" if count >= min_samples else "WARNING"
        print(f"  {label_name} ({label}): {count:,} {status}")
    
    # Check class balance
    if len(label_counts) > 1:
        counts = list(label_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nClass Balance:")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > max_imbalance:
            print(f"  WARNING: Class imbalance exceeds threshold ({max_imbalance})")
        else:
            print(f"  OK: Class balance is acceptable")
    
    # Content quality
    if content_lengths:
        avg_length = sum(content_lengths) / len(content_lengths)
        print(f"\nContent Quality:")
        print(f"  Average text length: {avg_length:.0f} characters")
        print(f"  Min text length: {min(content_lengths)}")
        print(f"  Max text length: {max(content_lengths)}")
    
    # Issues summary
    print(f"\nIssues Found:")
    for issue_type, issue_list in issues.items():
        print(f"  {issue_type}: {len(issue_list)}")
        if len(issue_list) <= 5:
            for item in issue_list:
                print(f"    - {item}")
    
    # Training readiness assessment
    print(f"\n{'='*60}")
    print("Training Readiness Assessment")
    print(f"{'='*60}")
    
    ready = True
    warnings = []
    
    # Check minimum samples per class
    for label, count in label_counts.items():
        if count < min_samples:
            ready = False
            label_name = {0: "severe", 1: "moderate", 2: "not depression"}.get(label, f"label_{label}")
            warnings.append(f"Class '{label_name}' ({label}) has only {count} samples (minimum: {min_samples})")
    
    # Check content quality
    if users_with_valid_content < valid_labeled_users * 0.8:
        warnings.append(f"Only {users_with_valid_content}/{valid_labeled_users} users have sufficient content")
    
    # Check class balance
    if len(label_counts) > 1:
        counts = list(label_counts.values())
        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        if imbalance_ratio > max_imbalance:
            warnings.append(f"Class imbalance ratio ({imbalance_ratio:.2f}) exceeds threshold ({max_imbalance})")
    
    if ready and not warnings:
        print("OK: Dataset is READY for training!")
    else:
        print("WARNING: Dataset has issues that should be addressed:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Save validation report
    validation_report = {
        "total_users": total_users,
        "valid_labeled_users": valid_labeled_users,
        "users_with_valid_content": users_with_valid_content,
        "label_distribution": dict(label_counts),
        "content_statistics": {
            "average_length": sum(content_lengths) / len(content_lengths) if content_lengths else 0,
            "min_length": min(content_lengths) if content_lengths else 0,
            "max_length": max(content_lengths) if content_lengths else 0
        },
        "issues": {k: len(v) for k, v in issues.items()},
        "training_ready": ready,
        "warnings": warnings
    }
    
    report_file = output_path / "validation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved validation report to: {report_file}")
    
    # Optionally save filtered dataset
    if all_valid_users:
        filtered_file = output_path / "filtered_labeled_data.json"
        with open(filtered_file, 'w', encoding='utf-8') as f:
            json.dump(all_valid_users, f, indent=2, ensure_ascii=False)
        print(f"Saved filtered dataset ({len(all_valid_users):,} users) to: {filtered_file}")
    
    return validation_report

def main():
    """Main execution function."""
    print("="*60)
    print("Script 3: Validate Labeled Data")
    print("="*60)
    
    try:
        config = load_config(str(Path(__file__).parent / "labeling_config.json"))
        agg_packet_dir = config['paths']['agg_packet']
        output_dir = config['paths']['output_dir']
        validation_config = config['validation']
        
        report = validate_labeled_data(agg_packet_dir, output_dir, validation_config)
        
        print(f"\n{'='*60}")
        print("SUCCESS: Validation complete!")
        print(f"{'='*60}")
        print(f"Next step: Run 'prepare_training_dataset.py'")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

