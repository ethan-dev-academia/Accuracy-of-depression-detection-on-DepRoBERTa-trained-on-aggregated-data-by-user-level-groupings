"""
Add true labels from RMH dataset CSV files to Reddit user JSON files.

The RMH dataset is expected to be at: F:/DATA STORAGE/RMH Dataset
The Reddit JSON files are at: F:/DATA STORAGE/AGG_PACKET

This script:
1. Loads username -> label mappings from RMH CSV files
2. Matches usernames in the Reddit JSON files to RMH labels
3. Adds a 'label' field to each user record in the JSON files
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Paths
RMH_DATASET_DIR = r"F:/DATA STORAGE/RMH Dataset"
AGG_PACKET_DIR = r"F:/DATA STORAGE/AGG_PACKET"

# Label mapping (based on model config: 0=severe, 1=moderate, 2=not depression)
# Adjust these mappings based on how labels appear in your RMH CSV files
LABEL_MAPPING = {
    "severe": 0,
    "moderate": 1,
    "not depression": 2,
    "not_depression": 2,
    "notdepression": 2,
    "normal": 2,
    "control": 2,
    # Add more mappings as needed
}

def load_rmh_labels(rmh_dir, username_col=1, label_col=0):
    """
    Load username -> label mappings from RMH CSV files.
    
    The RMH dataset uses subreddit membership as a proxy for depression labels:
    - 'depression' subreddit → severe (0)
    - 'anxiety' subreddit → moderate (1)
    - Other subreddits → not depression (2) or unlabeled
    
    Args:
        rmh_dir: Path to RMH dataset directory
        username_col: Column index containing usernames (default: 1, second column)
        label_col: Column index containing labels (default: 0, first column) - NOT USED
    
    Returns:
        dict: {username: label_int}
    """
    rmh_path = Path(rmh_dir)
    if not rmh_path.exists():
        raise FileNotFoundError(f"RMH dataset directory not found: {rmh_dir}")
    
    csv_files = list(rmh_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {rmh_dir}")
    
    print(f"Found {len(csv_files)} CSV file(s) in RMH dataset", flush=True)
    print(f"Processing files (this may take a few minutes)...", flush=True)
    
    # Subreddit to label mapping
    SUBREDDIT_LABEL_MAP = {
        'depression': 0,  # severe
        'anxiety': 1,  # moderate
        'ptsd': 1,  # moderate
        # Other subreddits (addiction, adhd, alcoholism, autism) can be mapped to 2 (not depression)
        # or left unlabeled - adjust as needed
        'addiction': 2,  # not depression (or could be 1 for moderate)
        'adhd': 2,  # not depression (or could be 1 for moderate)
        'alcoholism': 2,  # not depression (or could be 1 for moderate)
        'autism': 2,  # not depression (or could be 1 for moderate)
    }
    
    username_to_label = {}
    
    # Filter to only process relevant files (depression, anxiety, etc.)
    relevant_files = [f for f in csv_files if any(term in f.name.lower() for term in ['depression', 'anxiety'])]
    
    if relevant_files:
        print(f"Filtering to {len(relevant_files)} relevant files (depression/anxiety subreddits)", flush=True)
        csv_files = relevant_files
    
    processed = 0
    for idx, csv_file in enumerate(csv_files, 1):
        try:
            print(f"[{idx}/{len(csv_files)}] Loading labels from: {csv_file.name}", flush=True)
            
            # First, read a tiny sample to identify columns (very fast)
            print(f"  Identifying columns...", flush=True)
            sample_df = pd.read_csv(csv_file, nrows=1)
            cols = sample_df.columns.tolist()
            
            # Identify columns we need
            username_col_name = 'author' if 'author' in cols else (cols[1] if len(cols) > 1 else None)
            subreddit_col_name = 'subreddit' if 'subreddit' in cols else None
            
            if username_col_name is None:
                print(f"  Warning: Could not find username column, skipping", flush=True)
                continue
            
            # Determine label from filename first (faster than reading full file)
            filename_lower = csv_file.name.lower()
            label_int = None
            if 'depression' in filename_lower:
                label_int = 0
            elif 'anxiety' in filename_lower:
                label_int = 1
            
            # If not found in filename, check subreddit (requires reading file)
            if label_int is None:
                if subreddit_col_name:
                    # Read just one row to get subreddit
                    print(f"  Checking subreddit column...", flush=True)
                    sample_row = pd.read_csv(csv_file, nrows=1, usecols=[subreddit_col_name])
                    if len(sample_row) > 0:
                        subreddit_value = str(sample_row[subreddit_col_name].iloc[0]).lower()
                        label_int = SUBREDDIT_LABEL_MAP.get(subreddit_value)
                
                if label_int is None:
                    print(f"  Warning: Cannot determine label, skipping", flush=True)
                    continue
            
            print(f"  Label: {label_int} ({'severe' if label_int == 0 else 'moderate' if label_int == 1 else 'not depression'})", flush=True)
            print(f"  Reading usernames from column: '{username_col_name}'", flush=True)
            
            # Read usernames in chunks to avoid memory issues
            chunk_size = 10000
            total_read = 0
            valid_count = 0
            
            print(f"  Processing chunks of {chunk_size} rows...", flush=True)
            for chunk_idx, chunk in enumerate(pd.read_csv(csv_file, usecols=[username_col_name], chunksize=chunk_size), 1):
                # Extract username -> label mappings (vectorized - much faster!)
                # Clean usernames
                usernames = chunk[username_col_name].astype(str).str.strip()
                # Filter out invalid usernames
                valid_mask = (usernames != 'nan') & (usernames != '') & (usernames.notna())
                valid_usernames = usernames[valid_mask]
                
                # Assign labels (last occurrence wins if duplicates)
                for username in valid_usernames:
                    username_to_label[username] = label_int
                
                total_read += len(chunk)
                valid_count += len(valid_usernames)
                
                # Print progress every chunk
                print(f"  [Chunk {chunk_idx}] Processed {total_read:,} rows, {valid_count:,} valid usernames...", flush=True)
            
            print(f"  [OK] Loaded {valid_count:,} labels from {csv_file.name} (label={label_int})", flush=True)
            processed += 1
            
        except Exception as e:
            print(f"  [ERROR] Error reading {csv_file.name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}", flush=True)
    print(f"Successfully processed {processed}/{len(csv_files)} files", flush=True)
    print(f"Total unique username->label mappings: {len(username_to_label):,}", flush=True)
    
    # Show label distribution
    label_counts = defaultdict(int)
    for label in username_to_label.values():
        label_counts[label] += 1
    
    print("\nLabel distribution:", flush=True)
    for label, count in sorted(label_counts.items()):
        label_name = {0: "severe", 1: "moderate", 2: "not depression"}.get(label, f"unknown({label})")
        print(f"  {label_name} ({label}): {count:,}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    return username_to_label


def add_labels_to_json(agg_packet_dir, username_to_label, output_suffix="_labeled"):
    """
    Add labels to Reddit user JSON files.
    
    Args:
        agg_packet_dir: Path to AGG_PACKET directory
        username_to_label: dict mapping username -> label_int
        output_suffix: Suffix to add to output filenames (default: "_labeled")
    """
    agg_path = Path(agg_packet_dir)
    if not agg_path.exists():
        raise FileNotFoundError(f"AGG_PACKET directory not found: {agg_packet_dir}")
    
    json_files = sorted(agg_path.glob("reddit_user_analysis_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {agg_packet_dir}")
    
    # Filter out files that already have "_labeled" in the name to avoid double-processing
    json_files = [f for f in json_files if "_labeled" not in f.stem]
    
    if not json_files:
        print("  Warning: All JSON files already have labels. Nothing to process.", flush=True)
        return
    
    print(f"  Filtered to {len(json_files)} unlabeled JSON file(s) (excluding _labeled files)", flush=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Processing {len(json_files)} JSON file(s)", flush=True)
    print(f"{'='*60}", flush=True)
    
    total_users = 0
    labeled_users = 0
    unmatched_users = set()
    
    # Collect all labeled users for consolidated output
    all_labeled_users = []
    
    # Build lowercase lookup once for case-insensitive matching (much faster)
    print(f"  Building case-insensitive lookup table...", flush=True)
    lowercase_lookup = {k.lower(): v for k, v in username_to_label.items()}
    print(f"  [OK] Lookup table ready ({len(lowercase_lookup):,} entries)", flush=True)
    
    for idx, json_file in enumerate(json_files, 1):
        print(f"\n[{idx}/{len(json_files)}] Processing: {json_file.name}", flush=True)
        
        # Load JSON
        print(f"  Loading JSON file...", flush=True)
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                records = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [ERROR] Corrupted JSON file: {e}", flush=True)
            print(f"  Skipping {json_file.name} due to JSON parsing error", flush=True)
            continue
        except Exception as e:
            print(f"  [ERROR] Error reading {json_file.name}: {e}", flush=True)
            print(f"  Skipping {json_file.name}", flush=True)
            continue
        
        if not isinstance(records, list):
            print(f"  Warning: Expected list, got {type(records)}, skipping", flush=True)
            continue
        
        print(f"  Found {len(records):,} user records", flush=True)
        print(f"  Matching usernames to labels...", flush=True)
        
        # Add labels
        processed_count = 0
        file_labeled_count = 0
        
        for record_idx, record in enumerate(records, 1):
            total_users += 1
            username = str(record.get("username", "")).strip()
            
            if not username:
                continue
            
            # Try to find label (direct match first - fastest)
            if username in username_to_label:
                record["label"] = username_to_label[username]
                labeled_users += 1
                file_labeled_count += 1
                # Add to consolidated collection
                all_labeled_users.append(record.copy())
            else:
                # Try case-insensitive match using lowercase lookup dict
                username_lower = username.lower()
                if username_lower in lowercase_lookup:
                    record["label"] = lowercase_lookup[username_lower]
                    labeled_users += 1
                    file_labeled_count += 1
                    # Add to consolidated collection
                    all_labeled_users.append(record.copy())
                else:
                    unmatched_users.add(username)
                    record["label"] = None  # Mark as unlabeled
            
            processed_count += 1
            
            # Print progress every 1000 records
            if processed_count % 1000 == 0:
                print(f"  Processed {processed_count:,}/{len(records):,} records, {file_labeled_count:,} matched so far...", flush=True)
        
        # Save labeled version
        print(f"  Saving labeled JSON file...", flush=True)
        output_file = json_file.parent / f"{json_file.stem}{output_suffix}{json_file.suffix}"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        labeled_count = sum(1 for r in records if r.get("label") is not None)
        print(f"  [OK] Total users: {len(records):,}", flush=True)
        print(f"  [OK] Labeled users: {labeled_count:,}", flush=True)
        print(f"  [OK] Unlabeled users: {len(records) - labeled_count:,}", flush=True)
        print(f"  [OK] Saved to: {output_file.name}", flush=True)
    
    # Save all labeled users to a consolidated file
    if all_labeled_users:
        print(f"\n{'='*60}", flush=True)
        print(f"Saving all {len(all_labeled_users):,} labeled users to consolidated file...", flush=True)
        consolidated_file = agg_path / "all_labeled_users_consolidated.json"
        with open(consolidated_file, "w", encoding="utf-8") as f:
            json.dump(all_labeled_users, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved consolidated file: {consolidated_file.name}", flush=True)
        
        # Show label distribution in consolidated file
        label_counts = defaultdict(int)
        for user in all_labeled_users:
            label = user.get("label")
            if label is not None:
                label_counts[label] += 1
        
        print(f"\nConsolidated file label distribution:", flush=True)
        for label, count in sorted(label_counts.items()):
            label_name = {0: "severe", 1: "moderate", 2: "not depression"}.get(label, f"unknown({label})")
            print(f"  {label_name} ({label}): {count:,}", flush=True)
    
    print(f"\n{'='*60}", flush=True)
    print("Summary", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total users processed: {total_users:,}", flush=True)
    print(f"Successfully labeled: {labeled_users:,}", flush=True)
    print(f"Unlabeled: {total_users - labeled_users:,}", flush=True)
    print(f"Match rate: {labeled_users/total_users*100:.1f}%" if total_users > 0 else "N/A", flush=True)
    
    if unmatched_users and len(unmatched_users) <= 20:
        print(f"\nSample unmatched usernames (first 20):", flush=True)
        for username in sorted(list(unmatched_users))[:20]:
            print(f"  - {username}", flush=True)
    elif unmatched_users:
        print(f"\nTotal unmatched usernames: {len(unmatched_users):,} (showing first 20)", flush=True)
        for username in sorted(list(unmatched_users))[:20]:
            print(f"  - {username}", flush=True)


def main():
    print("="*60, flush=True)
    print("Adding Labels from RMH Dataset to Reddit JSON Files", flush=True)
    print("="*60, flush=True)
    
    # Step 1: Load RMH labels
    print("\nStep 1: Loading labels from RMH dataset...", flush=True)
    try:
        username_to_label = load_rmh_labels(RMH_DATASET_DIR)
    except Exception as e:
        print(f"[ERROR] Error loading RMH labels: {e}", flush=True)
        print("\nPlease check:", flush=True)
        print(f"1. RMH dataset exists at: {RMH_DATASET_DIR}", flush=True)
        print("2. CSV files have username and label columns", flush=True)
        print("3. Adjust username_col and label_col parameters if needed", flush=True)
        return
    
    if not username_to_label:
        print("[ERROR] No labels loaded. Exiting.", flush=True)
        return
    
    print(f"[OK] Loaded {len(username_to_label):,} username->label mappings", flush=True)
    
    # Step 2: Add labels to JSON files
    print("\nStep 2: Adding labels to Reddit JSON files...", flush=True)
    try:
        add_labels_to_json(AGG_PACKET_DIR, username_to_label)
    except Exception as e:
        print(f"[ERROR] Error adding labels: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60, flush=True)
    print("[OK] Done! Check the *_labeled.json files in AGG_PACKET directory", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()

