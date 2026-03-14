# Labeling Pipeline - User Guide

This guide explains how to add labels from the RMH (Reddit Mental Health) dataset to your AGG_PACKET Reddit user data for machine learning training.

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

Run all steps automatically:

```bash
cd "CheckPoint-11/19"
python run_labeling_pipeline.py
```

This will:
1. Extract labels from RMH dataset
2. Match labels to your AGG_PACKET data
3. Validate the labeled data
4. Prepare training-ready dataset

### Option 2: Run Scripts Individually

If you prefer to run each step separately:

```bash
# Step 1: Extract labels from RMH dataset
python extract_rmh_labels.py

# Step 2: Match labels to AGG_PACKET JSON files
python match_labels_to_agg_packet.py

# Step 3: Validate labeled data
python validate_labeled_data.py

# Step 4: Prepare training dataset
python prepare_training_dataset.py
```

## Prerequisites

1. **Data Paths**: Make sure these directories exist:
   - `F:/DATA STORAGE/RMH Dataset` - RMH dataset CSV files
   - `F:\DATA STORAGE\AGG_PACKET` - Your Reddit user JSON files

2. **Python Packages**: Install required packages:
   ```bash
   pip install pandas
   pip install datasets  # Optional, for HuggingFace Dataset format
   ```

3. **Configuration**: The `labeling_config.json` file is already configured, but you can modify it if needed.

## What Each Script Does

### Script 1: `extract_rmh_labels.py`
- Scans all CSV files in the RMH dataset
- Extracts usernames from the `author` column
- Determines labels based on subreddit (depression=0, anxiety=1, etc.)
- Handles conflicts (if a user appears in multiple subreddits)
- **Output**: `rmh_username_labels.json` and `rmh_label_stats.json`

### Script 2: `match_labels_to_agg_packet.py`
- Loads the username-label mappings from Script 1
- Matches usernames to your AGG_PACKET JSON files
- Adds a `label` field to each user record
- Creates labeled versions of your JSON files (with `_labeled` suffix)
- **Output**: 
  - `*_labeled.json` files in AGG_PACKET directory
  - `all_labeled_users.json` (consolidated file)
  - `label_matching_report.json`

### Script 3: `validate_labeled_data.py`
- Checks data quality (valid labels, required fields)
- Analyzes label distribution
- Checks content quality (sufficient text per user)
- Assesses training readiness
- **Output**: 
  - `validation_report.json`
  - `filtered_labeled_data.json` (clean dataset)

### Script 4: `prepare_training_dataset.py`
- Aggregates user posts and comments into single text
- Creates training dataset format
- Splits into train/validation/test sets (stratified by label)
- Saves in multiple formats (JSON, HuggingFace Dataset)
- **Output**:
  - `training_dataset.json` (full dataset)
  - `train.json`, `val.json`, `test.json` (splits)
  - `training_dataset_stats.json`

## Output Files

After running the pipeline, you'll find outputs in:
```
F:\DATA STORAGE\AGG_PACKET\labeling_outputs\
```

Key files:
- `rmh_username_labels.json` - Username to label mappings
- `all_labeled_users.json` - All users with labels
- `validation_report.json` - Data quality report
- `train.json`, `val.json`, `test.json` - Training splits
- `training_dataset_stats.json` - Dataset statistics

## Label Mapping

Labels are assigned based on subreddit membership:
- **0 (severe)**: Users from `depression` subreddit
- **1 (moderate)**: Users from `anxiety` or `ptsd` subreddits
- **2 (not depression)**: Users from other subreddits (adhd, addiction, etc.)

If a user appears in multiple subreddits, priority is: depression (0) > anxiety/ptsd (1) > other (2)

## Configuration

Edit `labeling_config.json` to customize:
- Data paths
- Label mappings
- Validation thresholds
- Train/val/test split ratios
- Output formats

## Troubleshooting

### "RMH dataset directory not found"
- Check that `F:/DATA STORAGE/RMH Dataset` exists
- Update path in `labeling_config.json` if different

### "No CSV files found"
- Make sure CSV files are in the RMH dataset directory
- Check file permissions

### "No labeled JSON files found"
- Run Script 1 and Script 2 first
- Check that AGG_PACKET directory contains `reddit_user_analysis_*.json` files

### Low match rate (<50%)
- Check if usernames in RMH dataset match those in AGG_PACKET
- Verify username format consistency
- Check for case sensitivity issues

### "Class imbalance exceeds threshold"
- Some labels have many more samples than others
- Consider:
  - Collecting more data for underrepresented classes
  - Using class weights during training
  - Adjusting `max_class_imbalance_ratio` in config

## Utility Scripts

### `inspect_labeled_data.py`
Quick inspection of labeled data:
```bash
python inspect_labeled_data.py
```

Shows:
- Label distribution
- Sample records from each class
- Basic statistics

## Next Steps

After labeling is complete:

1. **Review Validation Report**: Check `validation_report.json` for data quality
2. **Check Statistics**: Review `training_dataset_stats.json` for dataset info
3. **Start Training**: Use `train.json`, `val.json`, `test.json` for model training

## Example Usage

```bash
# Navigate to script directory
cd "F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\CheckPoint-11\19"

# Run complete pipeline
python run_labeling_pipeline.py

# Or run individually
python extract_rmh_labels.py
python match_labels_to_agg_packet.py
python validate_labeled_data.py
python prepare_training_dataset.py

# Inspect results
python inspect_labeled_data.py
```

## Support

If you encounter issues:
1. Check the error messages - they usually indicate what's wrong
2. Verify data paths in `labeling_config.json`
3. Check that input files exist and are readable
4. Review the output reports for warnings




