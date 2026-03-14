"""Explain the complete labeling pipeline and what each script does."""
from pathlib import Path

print("="*80)
print("LABELING PIPELINE - COMPLETE EXPLANATION")
print("="*80)

scripts_dir = Path(__file__).parent

print("\n" + "="*80)
print("OVERVIEW: 4 Main Scripts + Configuration")
print("="*80)
print("""
The labeling process uses 4 main scripts that run in sequence:
1. extract_rmh_labels.py      - Extracts labels from RMH dataset
2. match_labels_to_agg_packet.py - Matches labels to your data
3. validate_labeled_data.py   - Validates the labeled data
4. prepare_training_dataset.py - Creates training-ready format

Plus a master script:
5. run_labeling_pipeline.py    - Runs all scripts automatically
""")

print("\n" + "="*80)
print("SCRIPT 1: extract_rmh_labels.py")
print("="*80)
print("""
PURPOSE: Extract username-to-label mappings from RMH CSV files

WHAT IT DOES:
1. Scans all CSV files in: F:/DATA STORAGE/RMH Dataset
2. Reads the 'author' column (usernames) from each CSV
3. Determines label based on:
   - Filename (e.g., "depression_*.csv" -> label 0)
   - Subreddit column (e.g., "depression" -> label 0)
4. Creates mapping: {username: label}
5. Handles conflicts (if user in multiple subreddits, uses priority)

LABEL MAPPING:
- 'depression' subreddit -> Label 0 (severe)
- 'anxiety', 'ptsd' subreddits -> Label 1 (moderate)
- Other subreddits -> Label 2 (not depression) or skipped

OUTPUT FILES:
- rmh_username_labels.json: {username: label} dictionary
- rmh_label_stats.json: Statistics about extraction

RESULT: 172,354 username-label mappings extracted
""")

print("\n" + "="*80)
print("SCRIPT 2: match_labels_to_agg_packet.py")
print("="*80)
print("""
PURPOSE: Match extracted labels to your AGG_PACKET JSON files

WHAT IT DOES:
1. Loads rmh_username_labels.json from Script 1
2. Finds all JSON files in: F:\\DATA STORAGE\\AGG_PACKET
3. For each JSON file:
   - Loads user records
   - For each user, looks up username in label mappings
   - Adds 'label' field to user record:
     * If found: "label": 0 or 1
     * If not found: "label": None
4. Saves new files with "_labeled" suffix
5. Creates consolidated file with all labeled users

MATCHING STRATEGY:
- Case-insensitive matching (username.lower())
- Direct match first, then case-insensitive fallback
- Handles missing usernames gracefully

OUTPUT FILES:
- *_labeled.json files in AGG_PACKET directory
- all_labeled_users.json: Consolidated file with all labeled users
- label_matching_report.json: Matching statistics

RESULT: 147,130 users labeled out of 728,921 total (20.2% match rate)
""")

print("\n" + "="*80)
print("SCRIPT 3: validate_labeled_data.py")
print("="*80)
print("""
PURPOSE: Validate the labeled dataset for training readiness

WHAT IT DOES:
1. Loads labeled JSON files (or all_labeled_users.json)
2. Checks data quality:
   - Valid label values (0, 1, 2, or None)
   - Required fields present
   - Content quality (sufficient text length)
3. Analyzes label distribution:
   - Counts per label
   - Class balance ratio
   - Checks minimum samples per class
4. Assesses training readiness:
   - Sufficient examples per class?
   - Good class balance?
   - Quality content?

OUTPUT FILES:
- validation_report.json: Detailed validation results
- filtered_labeled_data.json: Clean dataset (optional)

RESULT: Validated 175,502 labeled users, 54,683 with sufficient content
""")

print("\n" + "="*80)
print("SCRIPT 4: prepare_training_dataset.py")
print("="*80)
print("""
PURPOSE: Create training-ready dataset from labeled data

WHAT IT DOES:
1. Loads labeled users (from all_labeled_users.json or labeled files)
2. Aggregates user content:
   - Combines all posts (title + content)
   - Combines all comments
   - Creates single text representation per user
3. Filters users:
   - Only users with valid labels (0 or 1)
   - Only users with sufficient content (min length)
4. Creates train/val/test splits:
   - Stratified by label (maintains class distribution)
   - Default: 80% train, 10% val, 10% test
5. Saves in multiple formats:
   - JSON format (train.json, val.json, test.json)
   - HuggingFace Dataset format (train_hf/, val_hf/, test_hf/)

OUTPUT FILES:
- training_dataset.json: Full dataset
- train.json, val.json, test.json: Split datasets
- training_dataset_stats.json: Dataset statistics
- HuggingFace format directories

RESULT: 34,172 training examples ready for model training
""")

print("\n" + "="*80)
print("MASTER SCRIPT: run_labeling_pipeline.py")
print("="*80)
print("""
PURPOSE: Orchestrate the entire pipeline

WHAT IT DOES:
1. Runs all 4 scripts in sequence
2. Handles errors and dependencies
3. Provides progress updates
4. Generates summary report

USAGE:
  python run_labeling_pipeline.py

This runs all steps automatically without manual intervention.
""")

print("\n" + "="*80)
print("CONFIGURATION: labeling_config.json")
print("="*80)
print("""
Contains all settings:
- Data paths (RMH dataset, AGG_PACKET)
- Label mappings (subreddit -> label)
- Validation thresholds
- Training split ratios
- Output formats

This file is read by all scripts to get configuration.
""")

print("\n" + "="*80)
print("DATA FLOW DIAGRAM")
print("="*80)
print("""
RMH Dataset (CSV files)
    |
    v
[Script 1: extract_rmh_labels.py]
    |
    v
rmh_username_labels.json (username -> label mappings)
    |
    v
[Script 2: match_labels_to_agg_packet.py]
    |
    v
AGG_PACKET JSON files + Labels
    |
    v
all_labeled_users.json (consolidated)
    |
    v
[Script 3: validate_labeled_data.py]
    |
    v
Validation Report + Filtered Data
    |
    v
[Script 4: prepare_training_dataset.py]
    |
    v
train.json, val.json, test.json (READY FOR TRAINING!)
""")

print("\n" + "="*80)
print("UTILITY SCRIPTS")
print("="*80)
print("""
Additional helper scripts:

- inspect_labeled_data.py: Quick inspection of labeled data
- show_examples.py: Display sample records
- explain_score.py: Explain what 'score' means
- show_labels.py: Show where labels are stored
- find_label_in_file.py: Help find label field in JSON

These are for exploration and debugging, not part of the main pipeline.
""")

print("\n" + "="*80)
print("FILE LOCATIONS")
print("="*80)
labeling_dir = Path(__file__).parent.parent / "labeling"
print(f"""
Labeling pipeline scripts: {labeling_dir}
Configuration: {labeling_dir / 'labeling_config.json'}
Output files: F:\\DATA STORAGE\\AGG_PACKET\\labeling_outputs\\
""")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
The labeling pipeline uses 4 main scripts:
1. Extract labels from RMH dataset
2. Match labels to your AGG_PACKET data
3. Validate the labeled data
4. Prepare training-ready dataset

Each script builds on the previous one's output, creating a complete
pipeline from raw data to training-ready dataset.
""")




