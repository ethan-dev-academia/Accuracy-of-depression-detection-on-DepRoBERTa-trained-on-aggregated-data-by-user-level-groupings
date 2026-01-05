# Data Selection and Filtering Process

This document clarifies how data was selected and filtered from the RMH (Reddit Mental Health) dataset for training the depression classification model.

---

## Original RMH Dataset

### Dataset Location
- **Path**: `F:/DATA STORAGE/RMH Dataset`
- **Format**: CSV files containing Reddit posts and comments
- **Structure**: Each row represents a post or comment with fields including:
  - `author` (username)
  - `subreddit` (subreddit name)
  - Post/comment content
  - Other metadata (timestamps, scores, etc.)

### Total Files in RMH Dataset
- **Total CSV files**: The extraction script scans all CSV files recursively in the RMH dataset directory
- **Note**: The exact number of total CSV files and total rows/posts in the complete RMH dataset is not explicitly tracked in the current implementation

---

## Filtering Criteria

### Subreddits Selected
The extraction process filters CSV files based on **filename keywords** to identify relevant mental health subreddits:

- **`depression`** → Label 0 (severe/depression)
- **`anxiety`** → Label 1 (moderate/anxiety/PTSD)
- **`ptsd`** → Label 1 (moderate/anxiety/PTSD)

### Filtering Process

1. **File-Level Filtering** (`extract_rmh_labels.py`, lines 46-52):
   - Scans all CSV files in the RMH dataset directory
   - Filters to files containing keywords: `'depression'`, `'anxiety'`, or `'ptsd'` in the filename
   - Only processes CSV files matching these criteria

2. **Username Extraction**:
   - From each filtered CSV file, extracts unique usernames from the `author` column
   - Each row in the CSV file represents a post or comment
   - Multiple posts/comments from the same user are collapsed to a single username-label mapping

### Filtering Results

**Files Processed**:
- The script processes only CSV files with relevant keywords in their filenames
- Files not containing 'depression', 'anxiety', or 'ptsd' in the filename are skipped

**Usernames Extracted**:
- **Total unique usernames extracted**: **172,354**
  - Label 0 (depression): **104,483 users**
  - Label 1 (anxiety/PTSD): **67,871 users**
  - Conflicts resolved: **6,661** (users appearing in multiple subreddits)

---

## Important Notes About "Posts" vs "Usernames"

### Data Granularity
The RMH dataset CSV files contain **individual posts and comments** (one per row), but the extraction process:

1. **Reads all rows** from filtered CSV files (each row = one post/comment)
2. **Extracts unique usernames** from the `author` column
3. **Maps usernames to labels** based on subreddit membership
4. **Result**: Username-level labels, not post-level labels

### Why Username-Level?
- The pipeline is designed for **user-level classification** (predicting depression based on a user's overall content)
- Training data aggregates all posts/comments per user into a single text representation
- Final training dataset: **34,172 user-level examples** (not post-level)

---

## Exact Post Counts (To Be Calculated)

To determine the exact number of:
- **Total posts in original RMH dataset**
- **Total posts remaining after filtering**

You would need to:

1. **Count total rows** in all CSV files in `F:/DATA STORAGE/RMH Dataset`
2. **Count rows** in filtered files (those with 'depression', 'anxiety', 'ptsd' in filename)
3. **Calculate the difference**

### Suggested Script to Calculate Post Counts

```python
import pandas as pd
from pathlib import Path
from collections import Counter

rmh_dir = Path(r'F:/DATA STORAGE/RMH Dataset')
csv_files = list(rmh_dir.rglob('*.csv'))

# Filter criteria
relevant_keywords = ['depression', 'anxiety', 'ptsd']
filtered_files = [f for f in csv_files if any(kw in f.name.lower() for kw in relevant_keywords)]

total_rows_all = 0
total_rows_filtered = 0

# Count rows in all files
print("Counting rows in all CSV files...")
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file, usecols=[0])  # Read only first column (fast)
        total_rows_all += len(df)
    except Exception as e:
        print(f"Error reading {csv_file.name}: {e}")

# Count rows in filtered files
print(f"\nCounting rows in filtered files (depression/anxiety/ptsd)...")
for csv_file in filtered_files:
    try:
        df = pd.read_csv(csv_file, usecols=[0])
        total_rows_filtered += len(df)
    except Exception as e:
        print(f"Error reading {csv_file.name}: {e}")

print(f"\n{'='*60}")
print(f"POST COUNT RESULTS")
print(f"{'='*60}")
print(f"Total CSV files: {len(csv_files)}")
print(f"Filtered CSV files: {len(filtered_files)}")
print(f"Total posts/comments (all files): {total_rows_all:,}")
print(f"Total posts/comments (filtered files): {total_rows_filtered:,}")
print(f"Posts filtered out: {total_rows_all - total_rows_filtered:,}")
print(f"Filter percentage: {100 * total_rows_filtered / total_rows_all:.2f}%")
```

---

## Current Statistics Available

Based on the extraction process, here's what we know:

### Extraction Statistics (from `rmh_label_stats.json`)
- **Total unique usernames**: 172,354
- **Files processed**: Number of filtered CSV files (exact count in stats file)
- **Label distribution**:
  - Depression (0): 104,483 users
  - Anxiety/PTSD (1): 67,871 users

### Training Dataset (after matching to AGG_PACKET)
- **Total labeled users matched**: 147,130 users (20.2% match rate from 728,921 total users in AGG_PACKET)
- **Final training examples**: 34,172 user-level examples
  - Training: 27,337
  - Validation: 3,416
  - Test: 3,419

---

## Summary

### What We Filter
- **Files**: Only CSV files with 'depression', 'anxiety', or 'ptsd' in filename
- **Subreddits**: Depression, anxiety, and PTSD subreddits only
- **Output**: Username-label mappings (not post-level)

### What We Track
- ✅ Unique usernames: **172,354**
- ✅ Label distribution
- ✅ Files processed
- ❌ Total post/comment counts (not currently tracked)

### For Your Documentation
To include exact post counts in your paper/documentation:

1. **Original dataset**: Need to count total rows in all RMH CSV files
2. **After filtering**: Need to count total rows in filtered CSV files only
3. **Note**: The final training uses **user-level** data (aggregated posts/comments per user), not post-level

**Recommended wording for documentation**:
> "We filtered the RMH dataset to include only subreddits related to depression, anxiety, and PTSD. From the filtered CSV files, we extracted 172,354 unique usernames (104,483 from depression subreddits, 67,871 from anxiety/PTSD subreddits). Each user's posts and comments were aggregated into a single text representation, resulting in 34,172 user-level training examples after matching with our AGG_PACKET dataset."

---

## Notes on "Suicidal" Subreddits

The current filtering does **not** explicitly include "suicidal" as a keyword. However:
- Some subreddits related to suicidal ideation may be included under 'depression' if they appear in files with 'depression' in the filename
- To explicitly include suicidal-related subreddits, you would need to add `'suicidal'` or `'suicidewatch'` to the `relevant_keywords` list in `extract_rmh_labels.py`

**Current keywords**: `['depression', 'anxiety', 'ptsd']`  
**Could be extended to**: `['depression', 'anxiety', 'ptsd', 'suicidal', 'suicidewatch']`

