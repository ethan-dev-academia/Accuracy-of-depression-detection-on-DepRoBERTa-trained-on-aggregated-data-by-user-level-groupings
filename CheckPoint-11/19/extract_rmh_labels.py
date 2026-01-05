"""
Script 1: Extract username-to-label mappings from RMH CSV files.

This script scans the RMH dataset CSV files, extracts usernames,
and determines labels based on subreddit membership.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import sys

def load_config(config_path="labeling_config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_rmh_labels(rmh_dir, label_mapping, output_dir):
    """
    Extract username -> label mappings from RMH CSV files.
    
    Args:
        rmh_dir: Path to RMH dataset directory
        label_mapping: Dict mapping subreddit names to label integers
        output_dir: Directory to save output files
    
    Returns:
        dict: {username: label_int}
    """
    rmh_path = Path(rmh_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not rmh_path.exists():
        raise FileNotFoundError(f"RMH dataset directory not found: {rmh_dir}")
    
    # Find all CSV files
    csv_files = list(rmh_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {rmh_dir}")
    
    print(f"Found {len(csv_files)} CSV file(s) in RMH dataset")
    print("Processing files (this may take a few minutes)...")
    
    # Filter to relevant files (depression, anxiety, ptsd)
    relevant_keywords = ['depression', 'anxiety', 'ptsd']
    relevant_files = [f for f in csv_files if any(kw in f.name.lower() for kw in relevant_keywords)]
    
    if relevant_files:
        print(f"Filtering to {len(relevant_files)} relevant files (depression/anxiety/ptsd subreddits)")
        csv_files = relevant_files
    
    username_to_label = {}
    label_conflicts = defaultdict(list)  # Track which files each username appears in
    processed_files = 0
    total_usernames = 0
    
    # Process each CSV file
    for idx, csv_file in enumerate(csv_files, 1):
        try:
            print(f"[{idx}/{len(csv_files)}] Processing: {csv_file.name}")
            
            # Read sample to identify columns
            sample_df = pd.read_csv(csv_file, nrows=1)
            cols = sample_df.columns.tolist()
            
            # Identify username and subreddit columns
            username_col = 'author' if 'author' in cols else (cols[1] if len(cols) > 1 else None)
            subreddit_col = 'subreddit' if 'subreddit' in cols else None
            
            if username_col is None:
                print(f"  Warning: Could not find username column, skipping")
                continue
            
            # Determine label from filename first
            filename_lower = csv_file.name.lower()
            label_int = None
            
            for subreddit_name, label_val in label_mapping.items():
                if subreddit_name in filename_lower:
                    label_int = label_val
                    break
            
            # If not found in filename, check subreddit column
            if label_int is None and subreddit_col:
                sample_row = pd.read_csv(csv_file, nrows=1, usecols=[subreddit_col])
                if len(sample_row) > 0:
                    subreddit_value = str(sample_row[subreddit_col].iloc[0]).lower()
                    label_int = label_mapping.get(subreddit_value)
            
            if label_int is None:
                print(f"  Warning: Cannot determine label, skipping")
                continue
            
            label_name = {0: "severe (depression)", 1: "moderate (anxiety/ptsd)", 2: "not depression"}.get(label_int, f"unknown({label_int})")
            print(f"  Label: {label_int} ({label_name})")
            print(f"  Reading usernames from column: '{username_col}'")
            
            # Read usernames in chunks
            chunk_size = 10000
            file_usernames = set()
            
            for chunk_idx, chunk in enumerate(pd.read_csv(csv_file, usecols=[username_col], chunksize=chunk_size), 1):
                usernames = chunk[username_col].astype(str).str.strip()
                valid_mask = (usernames != 'nan') & (usernames != '') & (usernames.notna())
                valid_usernames = usernames[valid_mask].unique()
                
                for username in valid_usernames:
                    username_lower = username.lower()
                    file_usernames.add(username)
                    
                    # Handle conflicts: keep highest priority (lowest number)
                    if username_lower in username_to_label:
                        old_label = username_to_label[username_lower]
                        if label_int < old_label:  # Lower number = higher priority
                            username_to_label[username_lower] = label_int
                            label_conflicts[username_lower].append((csv_file.name, old_label, label_int))
                    else:
                        username_to_label[username_lower] = label_int
                
                if chunk_idx % 10 == 0:
                    print(f"    Processed {chunk_idx * chunk_size:,} rows...")
            
            print(f"  [OK] Extracted {len(file_usernames):,} usernames from {csv_file.name}")
            total_usernames += len(file_usernames)
            processed_files += 1
            
        except Exception as e:
            print(f"  [ERROR] Error processing {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Extraction Complete")
    print(f"{'='*60}")
    print(f"Processed {processed_files}/{len(csv_files)} files")
    print(f"Total unique usernames: {len(username_to_label):,}")
    
    # Calculate label distribution
    label_counts = Counter(username_to_label.values())
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        label_name = {0: "severe (depression)", 1: "moderate (anxiety/ptsd)", 2: "not depression"}.get(label, f"unknown({label})")
        print(f"  {label_name} ({label}): {count:,}")
    
    if label_conflicts:
        print(f"\nResolved {len(label_conflicts):,} label conflicts")
    
    # Save outputs
    output_file = output_path / "rmh_username_labels.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(username_to_label, f, indent=2, ensure_ascii=False)
    print(f"\nSaved username-label mappings to: {output_file}")
    
    # Save statistics
    stats = {
        "total_usernames": len(username_to_label),
        "label_distribution": dict(label_counts),
        "files_processed": processed_files,
        "total_files": len(csv_files),
        "conflicts_resolved": len(label_conflicts),
        "conflict_details": {k: v for k, v in list(label_conflicts.items())[:100]}  # Sample
    }
    
    stats_file = output_path / "rmh_label_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved statistics to: {stats_file}")
    
    return username_to_label

def main():
    """Main execution function."""
    print("="*60)
    print("Script 1: Extract RMH Labels")
    print("="*60)
    
    try:
        config = load_config()
        rmh_dir = config['paths']['rmh_dataset']
        output_dir = config['paths']['output_dir']
        label_mapping = config['label_mapping']
        
        username_to_label = extract_rmh_labels(rmh_dir, label_mapping, output_dir)
        
        print(f"\n{'='*60}")
        print("SUCCESS: Label extraction complete!")
        print(f"{'='*60}")
        print(f"Next step: Run 'match_labels_to_agg_packet.py'")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

