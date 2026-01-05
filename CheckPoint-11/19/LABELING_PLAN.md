# Plan: Adding Labels from RMH Dataset to AGG_PACKET Data

## Overview
This plan outlines the scripts needed to extract labels from the RMH (Reddit Mental Health) dataset and attach them to the AGG_PACKET Reddit user data for machine learning training.

## Data Flow
```
RMH Dataset (CSV files) → Extract Usernames & Labels → Match with AGG_PACKET (JSON) → Labeled Training Data
```

---

## Script 1: `extract_rmh_labels.py`
**Purpose**: Extract username-to-label mappings from RMH CSV files

### Inputs
- RMH dataset directory: `F:/DATA STORAGE/RMH Dataset`
- Configuration for label mapping rules

### Process
1. **Scan RMH Dataset**
   - Find all CSV files recursively in RMH directory
   - Filter relevant files (depression, anxiety, ptsd subreddits)
   - Report total files found

2. **Extract Username-Label Mappings**
   - For each CSV file:
     - Read CSV file (in chunks for large files)
     - Identify columns: `author` (username), `subreddit`
     - Determine label from:
       - Filename (e.g., "depression_*.csv" → label 0)
       - Subreddit column value (e.g., "depression" → label 0)
     - Extract all unique usernames from `author` column
     - Map each username to its label

3. **Handle Label Conflicts**
   - If username appears in multiple files:
     - Priority order: depression (0) > anxiety/ptsd (1) > other (2)
     - Keep highest priority label (lowest number)
   - Track conflicts for reporting

4. **Data Cleaning**
   - Remove invalid usernames (empty, "nan", etc.)
   - Normalize usernames (strip whitespace, handle case)
   - Create case-insensitive lookup dictionary

### Outputs
- `rmh_username_labels.json`: Dictionary mapping `{username: label_int}`
- `rmh_label_stats.json`: Statistics about labels extracted
  - Total unique usernames
  - Label distribution (counts per label)
  - Files processed
  - Conflicts resolved

### Configuration
- Label mapping rules:
  - `depression` → 0 (severe)
  - `anxiety` → 1 (moderate)
  - `ptsd` → 1 (moderate)
  - `addiction`, `adhd`, `alcoholism`, `autism` → 2 (not depression) or None
- Which subreddits to include/exclude
- Chunk size for reading large CSV files

---

## Script 2: `match_labels_to_agg_packet.py`
**Purpose**: Match extracted labels to AGG_PACKET JSON files and add label field

### Inputs
- `rmh_username_labels.json` (from Script 1)
- AGG_PACKET directory: `F:\DATA STORAGE\AGG_PACKET`
- Output configuration (overwrite vs. create new files)

### Process
1. **Load Label Mappings**
   - Load `rmh_username_labels.json`
   - Create case-insensitive lookup dictionary
   - Report total labels available

2. **Process AGG_PACKET JSON Files**
   - Find all `reddit_user_analysis_*.json` files
   - Filter out already-labeled files (those with `_labeled` suffix)
   - For each JSON file:
     - Load JSON records
     - For each user record:
       - Extract `username` field
       - Look up username in label mappings (case-insensitive)
       - Add `label` field to record:
         - If found: `"label": <label_int>`
         - If not found: `"label": None`
     - Save labeled version

3. **Matching Strategy**
   - Direct match (exact username)
   - Case-insensitive match (username.lower())
   - Handle edge cases:
     - Missing username field
     - Username is None or empty
     - Special characters in usernames

4. **Output Files**
   - Option A: Create new files with `_labeled` suffix
   - Option B: Overwrite original files (with backup)
   - Create consolidated file: `all_labeled_users.json` (only users with labels)

### Outputs
- Labeled JSON files (original name + `_labeled` suffix)
- `all_labeled_users.json`: Consolidated file with only labeled users
- `label_matching_report.json`: Statistics
  - Total users processed
  - Users matched (with labels)
  - Users unmatched (no label)
  - Match rate percentage
  - Label distribution in matched users

### Configuration
- Output mode: create new files vs. overwrite
- Whether to include unlabeled users in consolidated file
- Minimum match rate threshold (warning if too low)

---

## Script 3: `validate_labeled_data.py`
**Purpose**: Validate the labeled dataset for training readiness

### Inputs
- Labeled JSON files from Script 2
- Training requirements (minimum samples per class, etc.)

### Process
1. **Data Quality Checks**
   - Verify all required fields exist (`username`, `posts`, `comments`, `label`)
   - Check for null/invalid labels
   - Verify label values are valid (0, 1, 2, or None)
   - Check for empty posts/comments arrays

2. **Label Distribution Analysis**
   - Count users per label
   - Calculate class balance/imbalance
   - Identify underrepresented classes
   - Check for sufficient samples per class (minimum threshold)

3. **Content Quality Checks**
   - Users with no valid text content (all posts/comments removed/deleted)
   - Users with very short content (below minimum threshold)
   - Users with only removed/deleted content

4. **Training Readiness Assessment**
   - Total labeled examples available
   - Class distribution balance
   - Content quality metrics
   - Recommendations for training

### Outputs
- `validation_report.json`: Detailed validation results
- `training_readiness_summary.txt`: Human-readable summary
- `filtered_labeled_data.json`: Clean dataset ready for training (optional)
  - Only users with valid labels
  - Only users with sufficient content
  - Balanced subset (if needed)

### Configuration
- Minimum samples per class for training
- Minimum content length per user
- Class balance thresholds
- Whether to create filtered dataset

---

## Script 4: `prepare_training_dataset.py`
**Purpose**: Convert labeled JSON data into format ready for model training

### Inputs
- Labeled JSON files (from Script 2 or filtered from Script 3)
- Training configuration

### Process
1. **Load Labeled Data**
   - Load all labeled JSON files
   - Or load filtered dataset from Script 3
   - Filter to only users with valid labels (not None)

2. **Aggregate User Content**
   - For each user:
     - Combine all posts (title + content)
     - Combine all comments
     - Create aggregated text representation
     - Keep track of segments (for analysis)

3. **Create Training Dataset**
   - Format: List of dictionaries with:
     - `text`: Aggregated user content
     - `label`: Integer label (0, 1, or 2)
     - `user_id`: Username
     - `segments`: List of text segments (optional)
   - Or create HuggingFace Dataset format

4. **Split Data** (Optional)
   - Train/validation/test splits
   - Stratified by label to maintain class distribution
   - Save split indices or separate files

5. **Export Formats**
   - JSON format (for custom training)
   - HuggingFace Dataset format (for transformers)
   - CSV format (for simple models)
   - Parquet format (for efficient storage)

### Outputs
- `training_dataset.json`: Full labeled dataset
- `train.json`, `val.json`, `test.json`: Split datasets (if splitting)
- `training_dataset_stats.json`: Dataset statistics
  - Total examples
  - Label distribution
  - Average text length
  - Split sizes

### Configuration
- Text aggregation method (simple join vs. structured)
- Train/val/test split ratios
- Random seed for reproducibility
- Output format(s) to generate

---

## Script 5: `run_labeling_pipeline.py` (Master Script)
**Purpose**: Orchestrate the entire labeling pipeline

### Process
1. **Run Script 1**: Extract RMH labels
   - Check if output already exists (skip if present)
   - Report progress

2. **Run Script 2**: Match labels to AGG_PACKET
   - Check dependencies (Script 1 output)
   - Report match statistics

3. **Run Script 3**: Validate labeled data
   - Check if validation passed
   - Warn if issues found
   - Optionally create filtered dataset

4. **Run Script 4**: Prepare training dataset
   - Generate final training-ready format
   - Report final statistics

5. **Generate Summary Report**
   - Pipeline execution summary
   - Data flow statistics
   - Final dataset characteristics
   - Ready for training confirmation

### Outputs
- `pipeline_execution_log.txt`: Execution log
- `pipeline_summary.json`: Complete pipeline summary
- All outputs from individual scripts

### Configuration
- Which steps to run (can skip if outputs exist)
- Error handling (stop on error vs. continue)
- Logging level

---

## Additional Utility Scripts

### Script 6: `inspect_labeled_data.py`
**Purpose**: Quick inspection of labeled data

### Process
- Load labeled JSON files
- Display sample records
- Show label distribution
- Check data quality metrics
- Interactive exploration

---

### Script 7: `fix_label_issues.py`
**Purpose**: Fix common labeling issues

### Process
- Detect duplicate usernames with different labels
- Resolve conflicts based on priority rules
- Fix invalid label values
- Clean up malformed records

---

## File Structure After Execution

```
F:\DATA STORAGE\AGG_PACKET\
├── reddit_user_analysis_*.json (original)
├── reddit_user_analysis_*_labeled.json (labeled versions)
├── all_labeled_users.json (consolidated)
└── labeling_outputs/
    ├── rmh_username_labels.json
    ├── rmh_label_stats.json
    ├── label_matching_report.json
    ├── validation_report.json
    ├── training_dataset.json
    ├── train.json
    ├── val.json
    ├── test.json
    └── training_dataset_stats.json
```

---

## Error Handling & Edge Cases

### Script 1 (Extract Labels)
- Handle corrupted CSV files
- Handle missing columns
- Handle encoding issues
- Handle very large files (memory management)
- Handle duplicate usernames across files

### Script 2 (Match Labels)
- Handle corrupted JSON files
- Handle missing username field
- Handle case sensitivity issues
- Handle special characters in usernames
- Handle users appearing in multiple JSON files

### Script 3 (Validate)
- Handle invalid label values
- Handle missing required fields
- Handle empty content
- Handle class imbalance warnings

### Script 4 (Prepare Training)
- Handle memory issues with large datasets
- Handle text encoding problems
- Handle very long text sequences
- Handle missing content

---

## Configuration Files

### `labeling_config.json`
```json
{
  "paths": {
    "rmh_dataset": "F:/DATA STORAGE/RMH Dataset",
    "agg_packet": "F:\\DATA STORAGE\\AGG_PACKET",
    "output_dir": "F:\\DATA STORAGE\\AGG_PACKET\\labeling_outputs"
  },
  "label_mapping": {
    "depression": 0,
    "anxiety": 1,
    "ptsd": 1,
    "addiction": 2,
    "adhd": 2,
    "alcoholism": 2,
    "autism": 2
  },
  "matching": {
    "case_sensitive": false,
    "priority_order": [0, 1, 2]
  },
  "validation": {
    "min_samples_per_class": 100,
    "min_content_length": 50,
    "max_class_imbalance_ratio": 10.0
  },
  "training_prep": {
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "random_seed": 42,
    "output_formats": ["json", "huggingface"]
  }
}
```

---

## Execution Order

1. **First Time Setup**
   ```
   Script 1 → Script 2 → Script 3 → Script 4
   ```

2. **Re-labeling (if RMH data updated)**
   ```
   Script 1 → Script 2 → Script 3 → Script 4
   ```

3. **Re-validation (if AGG_PACKET updated)**
   ```
   Script 2 → Script 3 → Script 4
   ```

4. **Just prepare training format**
   ```
   Script 4 (if labeled data already exists)
   ```

---

## Success Criteria

- [ ] All RMH CSV files processed successfully
- [ ] Username-label mappings extracted correctly
- [ ] Labels matched to AGG_PACKET users with >80% match rate
- [ ] All labeled users have valid label values (0, 1, or 2)
- [ ] Class distribution is acceptable for training
- [ ] Training dataset created in required format
- [ ] All statistics and reports generated
- [ ] Data ready for model training

---

## Testing Strategy

1. **Unit Tests**: Test each script with small sample data
2. **Integration Tests**: Test full pipeline with subset of data
3. **Validation Tests**: Verify label correctness on known examples
4. **Performance Tests**: Test with full dataset for memory/time issues

---

## Documentation Requirements

- README for each script (purpose, usage, parameters)
- Example usage commands
- Troubleshooting guide
- Configuration file documentation

