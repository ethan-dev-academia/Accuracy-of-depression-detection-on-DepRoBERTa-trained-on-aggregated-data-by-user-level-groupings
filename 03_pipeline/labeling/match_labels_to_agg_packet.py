"""
Script 2: Match extracted labels to AGG_PACKET JSON files.

This script loads username-label mappings and adds label fields
to user records in AGG_PACKET JSON files.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import sys

def load_config(config_path="labeling_config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def match_labels_to_agg_packet(agg_packet_dir, labels_file, output_dir, case_sensitive=False):
    """
    Match labels to AGG_PACKET JSON files.
    
    Args:
        agg_packet_dir: Path to AGG_PACKET directory
        labels_file: Path to rmh_username_labels.json
        output_dir: Directory to save outputs
        case_sensitive: Whether to match usernames case-sensitively
    
    Returns:
        dict: Matching statistics
    """
    agg_path = Path(agg_packet_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not agg_path.exists():
        raise FileNotFoundError(f"AGG_PACKET directory not found: {agg_packet_dir}")
    
    # Load label mappings
    print("Loading username-label mappings...")
    labels_path = Path(labels_file)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}\nRun 'extract_rmh_labels.py' first!")
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        username_to_label = json.load(f)
    
    print(f"Loaded {len(username_to_label):,} username-label mappings")
    
    # Create case-insensitive lookup if needed
    if not case_sensitive:
        lowercase_lookup = {k.lower(): v for k, v in username_to_label.items()}
        print("Created case-insensitive lookup table")
    
    # Find JSON files
    json_files = sorted(agg_path.glob("reddit_user_analysis_*.json"))
    json_files = [f for f in json_files if "_labeled" not in f.stem]
    
    if not json_files:
        print("No unlabeled JSON files found in AGG_PACKET directory")
        return
    
    print(f"\nFound {len(json_files)} unlabeled JSON file(s)")
    print("="*60)
    
    total_users = 0
    labeled_users = 0
    unmatched_users = set()
    all_labeled_users = []
    label_counts = Counter()
    
    # Process each JSON file
    for idx, json_file in enumerate(json_files, 1):
        print(f"\n[{idx}/{len(json_files)}] Processing: {json_file.name}")
        
        try:
            # Load JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [ERROR] Corrupted JSON file: {e}")
            print(f"  Skipping {json_file.name}")
            continue
        except Exception as e:
            print(f"  [ERROR] Error reading file: {e}")
            continue
        
        if not isinstance(records, list):
            print(f"  Warning: Expected list, got {type(records)}, skipping")
            continue
        
        print(f"  Found {len(records):,} user records")
        print(f"  Matching usernames to labels...")
        
        file_labeled_count = 0
        
        # Add labels to each record
        for record_idx, record in enumerate(records, 1):
            total_users += 1
            username = str(record.get("username", "")).strip()
            
            if not username:
                record["label"] = None
                continue
            
            # Try to find label
            label = None
            
            if case_sensitive:
                if username in username_to_label:
                    label = username_to_label[username]
            else:
                username_lower = username.lower()
                if username_lower in lowercase_lookup:
                    label = lowercase_lookup[username_lower]
            
            if label is not None:
                record["label"] = label
                labeled_users += 1
                file_labeled_count += 1
                label_counts[label] += 1
                all_labeled_users.append(record.copy())
            else:
                record["label"] = None
                unmatched_users.add(username)
            
            # Progress update
            if record_idx % 1000 == 0:
                print(f"    Processed {record_idx:,}/{len(records):,} records, {file_labeled_count:,} matched...")
        
        # Save labeled version
        output_file = json_file.parent / f"{json_file.stem}_labeled{json_file.suffix}"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        print(f"  [OK] Labeled users: {file_labeled_count:,}/{len(records):,}")
        print(f"  [OK] Saved to: {output_file.name}")
    
    # Save consolidated file with all labeled users
    if all_labeled_users:
        print(f"\n{'='*60}")
        print(f"Saving consolidated file with {len(all_labeled_users):,} labeled users...")
        consolidated_file = output_path / "all_labeled_users.json"
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(all_labeled_users, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved: {consolidated_file.name}")
    
    # Generate report
    match_rate = (labeled_users / total_users * 100) if total_users > 0 else 0
    
    report = {
        "total_users_processed": total_users,
        "users_labeled": labeled_users,
        "users_unlabeled": total_users - labeled_users,
        "match_rate_percent": round(match_rate, 2),
        "label_distribution": dict(label_counts),
        "files_processed": len(json_files),
        "sample_unmatched_usernames": sorted(list(unmatched_users))[:20]
    }
    
    report_file = output_path / "label_matching_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Matching Summary")
    print(f"{'='*60}")
    print(f"Total users processed: {total_users:,}")
    print(f"Successfully labeled: {labeled_users:,}")
    print(f"Unlabeled: {total_users - labeled_users:,}")
    print(f"Match rate: {match_rate:.1f}%")
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        label_name = {0: "severe (depression)", 1: "moderate (anxiety/ptsd)", 2: "not depression"}.get(label, f"unknown({label})")
        print(f"  {label_name} ({label}): {count:,}")
    print(f"\nSaved report to: {report_file}")
    
    return report

def main():
    """Main execution function."""
    print("="*60)
    print("Script 2: Match Labels to AGG_PACKET")
    print("="*60)
    
    try:
        config = load_config(str(Path(__file__).parent / "labeling_config.json"))
        agg_packet_dir = config['paths']['agg_packet']
        output_dir = config['paths']['output_dir']
        case_sensitive = config['matching']['case_sensitive']
        
        # Construct path to labels file
        labels_file = Path(output_dir) / "rmh_username_labels.json"
        
        report = match_labels_to_agg_packet(
            agg_packet_dir, 
            str(labels_file), 
            output_dir,
            case_sensitive
        )
        
        print(f"\n{'='*60}")
        print("SUCCESS: Label matching complete!")
        print(f"{'='*60}")
        print(f"Next step: Run 'validate_labeled_data.py'")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()




