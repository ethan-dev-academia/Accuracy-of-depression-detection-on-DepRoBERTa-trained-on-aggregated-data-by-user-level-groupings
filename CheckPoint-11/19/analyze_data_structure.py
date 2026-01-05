"""
Analyze the structure of both datasets to understand how to match labels accurately.

This script:
1. Inspects the RMH dataset structure (CSV files with usernames)
2. Inspects the AGG_PACKET structure (JSON files with user posts/comments)
3. Shows how usernames are stored in both
4. Provides recommendations for accurate label matching
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

# Paths
RMH_DATASET_DIR = Path(r"F:/DATA STORAGE/RMH Dataset")
AGG_PACKET_DIR = Path(r"F:\DATA STORAGE\AGG_PACKET")

print("="*80)
print("ANALYZING DATA STRUCTURE FOR LABEL MATCHING")
print("="*80)

# ============================================================================
# PART 1: Analyze RMH Dataset (Mental Health Dataset)
# ============================================================================
print("\n" + "="*80)
print("PART 1: RMH DATASET STRUCTURE (Source of Labels)")
print("="*80)

if not RMH_DATASET_DIR.exists():
    print(f"❌ RMH Dataset directory not found: {RMH_DATASET_DIR}")
    print("   Please check the path.")
else:
    csv_files = list(RMH_DATASET_DIR.rglob("*.csv"))
    print(f"\nFound {len(csv_files)} CSV file(s)")
    
    if csv_files:
        # Sample a few files to understand structure
        sample_files = csv_files[:5]
        print(f"\nAnalyzing {len(sample_files)} sample files:")
        
        all_usernames = set()
        subreddit_info = {}
        
        for csv_file in sample_files:
            print(f"\n  File: {csv_file.name}")
            try:
                # Read first few rows
                df_sample = pd.read_csv(csv_file, nrows=10)
                print(f"    Columns: {list(df_sample.columns)}")
                print(f"    Shape (sample): {df_sample.shape}")
                
                # Identify username column
                username_col = None
                if 'author' in df_sample.columns:
                    username_col = 'author'
                elif len(df_sample.columns) > 1:
                    username_col = df_sample.columns[1]  # Usually second column
                
                # Identify subreddit column
                subreddit_col = None
                if 'subreddit' in df_sample.columns:
                    subreddit_col = 'subreddit'
                
                print(f"    Username column: {username_col}")
                print(f"    Subreddit column: {subreddit_col}")
                
                if username_col:
                    # Read more rows to get usernames
                    df_usernames = pd.read_csv(csv_file, usecols=[username_col], nrows=1000)
                    usernames = df_usernames[username_col].dropna().astype(str).str.strip()
                    usernames = usernames[usernames != 'nan']
                    all_usernames.update(usernames)
                    print(f"    Sample usernames: {list(usernames.head(5))}")
                
                if subreddit_col:
                    df_subreddit = pd.read_csv(csv_file, usecols=[subreddit_col], nrows=1000)
                    subreddits = df_subreddit[subreddit_col].dropna().unique()
                    subreddit_info[csv_file.name] = list(subreddits)
                    print(f"    Subreddits: {list(subreddits)}")
                
            except Exception as e:
                print(f"    ❌ Error reading file: {e}")
        
        print(f"\n  Total unique usernames found (sample): {len(all_usernames)}")
        
        # Check for label-related files
        print(f"\n  Checking for label-related files:")
        label_keywords = ['depression', 'anxiety', 'ptsd', 'control', 'normal']
        for keyword in label_keywords:
            matching = [f.name for f in csv_files if keyword.lower() in f.name.lower()]
            if matching:
                print(f"    '{keyword}': {len(matching)} file(s) - {matching[:3]}")

# ============================================================================
# PART 2: Analyze AGG_PACKET Dataset (Your Reddit User Data)
# ============================================================================
print("\n" + "="*80)
print("PART 2: AGG_PACKET DATASET STRUCTURE (Target for Labels)")
print("="*80)

if not AGG_PACKET_DIR.exists():
    print(f"❌ AGG_PACKET directory not found: {AGG_PACKET_DIR}")
    print("   Please check the path.")
else:
    json_files = sorted(AGG_PACKET_DIR.glob("reddit_user_analysis_*.json"))
    # Filter out labeled files
    json_files = [f for f in json_files if "_labeled" not in f.stem]
    
    print(f"\nFound {len(json_files)} JSON file(s) (excluding _labeled files)")
    
    if json_files:
        # Sample the first file
        sample_file = json_files[0]
        print(f"\nAnalyzing sample file: {sample_file.name}")
        
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            if not isinstance(records, list):
                print(f"  ❌ Expected list, got {type(records)}")
            else:
                print(f"  Total records: {len(records)}")
                
                if records:
                    # Analyze first record
                    first_record = records[0]
                    print(f"\n  📋 Record structure:")
                    print(f"    Top-level keys: {list(first_record.keys())}")
                    
                    # Check username field
                    username = first_record.get('username') or first_record.get('user_id')
                    print(f"    Username field: {username}")
                    print(f"    Username type: {type(username)}")
                    
                    # Check for existing label
                    if 'label' in first_record:
                        print(f"    ⚠️  Label already exists: {first_record.get('label')}")
                    else:
                        print(f"    ✓ No label field (ready for labeling)")
                    
                    # Check posts/comments structure
                    posts = first_record.get('posts', [])
                    comments = first_record.get('comments', [])
                    print(f"    Posts: {len(posts)}")
                    print(f"    Comments: {len(comments)}")
                    
                    if posts:
                        print(f"    Post structure: {list(posts[0].keys()) if posts else 'N/A'}")
                    if comments:
                        print(f"    Comment structure: {list(comments[0].keys()) if comments else 'N/A'}")
                    
                    # Sample usernames
                    print(f"\n  Sample usernames from dataset:")
                    sample_usernames = [r.get('username') or r.get('user_id', 'N/A') for r in records[:10]]
                    for i, uname in enumerate(sample_usernames, 1):
                        print(f"    {i}. {uname}")
                    
                    # Check username format consistency
                    username_types = Counter(type(r.get('username') or r.get('user_id')).__name__ for r in records[:100])
                    print(f"\n  Username type distribution (first 100): {dict(username_types)}")
                    
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
            import traceback
            traceback.print_exc()

# ============================================================================
# PART 3: Recommendations for Accurate Label Matching
# ============================================================================
print("\n" + "="*80)
print("PART 3: RECOMMENDATIONS FOR ACCURATE LABEL MATCHING")
print("="*80)

print("""
Based on the analysis above, here's how to accurately add labels:

1. **Matching Strategy:**
   - Match usernames from RMH CSV files to usernames in AGG_PACKET JSON files
   - Use case-insensitive matching (Reddit usernames are case-sensitive but 
     sometimes stored inconsistently)
   - Handle whitespace and special characters

2. **Label Assignment:**
   - Labels are inferred from subreddit membership in RMH dataset:
     * 'depression' subreddit → label 0 (severe)
     * 'anxiety' subreddit → label 1 (moderate)  
     * Other subreddits → label 2 (not depression) or None (unlabeled)
   
3. **Accuracy Considerations:**
   - A user might appear in multiple CSV files (multiple subreddits)
   - Priority: depression (0) > anxiety (1) > other (2)
   - If a user appears in both depression and anxiety files, assign label 0 (severe)
   
4. **Implementation:**
   - Load all usernames from RMH CSV files with their inferred labels
   - Create a lookup dictionary: {username_lowercase: label}
   - For each user in AGG_PACKET, check if username exists in lookup
   - Add 'label' field to the JSON record

5. **Validation:**
   - Check match rate (how many users got labels)
   - Verify label distribution matches expectations
   - Handle edge cases (missing usernames, special characters, etc.)
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

