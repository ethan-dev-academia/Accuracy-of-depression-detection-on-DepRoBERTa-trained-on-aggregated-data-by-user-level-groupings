"""Generate comprehensive status report."""
import json
from pathlib import Path
from collections import Counter

print("="*80)
print("PROJECT STATUS REPORT")
print("="*80)

# Scripts directory
scripts_dir = Path(__file__).parent
output_dir = Path(r"F:\DATA STORAGE\AGG_PACKET\labeling_outputs")

print("\n" + "="*80)
print("1. SCRIPTS CREATED")
print("="*80)

scripts = sorted(scripts_dir.glob("*.py"))
print(f"\nTotal Python scripts: {len(scripts)}")
print("\nCore Pipeline Scripts:")
core_scripts = [
    "extract_rmh_labels.py",
    "match_labels_to_agg_packet.py", 
    "validate_labeled_data.py",
    "prepare_training_dataset.py",
    "run_labeling_pipeline.py"
]
for script in core_scripts:
    if (scripts_dir / script).exists():
        print(f"  [OK] {script}")
    else:
        print(f"  [MISSING] {script}")

print("\nUtility Scripts:")
utility_scripts = [
    "inspect_labeled_data.py",
    "show_examples.py",
    "explain_score.py",
    "show_labels.py",
    "find_label_in_file.py",
    "start_training.py"
]
for script in utility_scripts:
    if (scripts_dir / script).exists():
        print(f"  [OK] {script}")
    else:
        print(f"  [MISSING] {script}")

print("\n" + "="*80)
print("2. DATA STATUS")
print("="*80)

# Check output files
if output_dir.exists():
    json_files = list(output_dir.glob("*.json"))
    print(f"\nOutput files created: {len(json_files)}")
    
    # Key files
    key_files = {
        "rmh_username_labels.json": "RMH label mappings",
        "all_labeled_users.json": "Consolidated labeled users",
        "train.json": "Training split",
        "val.json": "Validation split",
        "test.json": "Test split",
        "training_dataset_stats.json": "Dataset statistics",
        "label_matching_report.json": "Matching report"
    }
    
    print("\nKey output files:")
    for filename, description in key_files.items():
        filepath = output_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024*1024)
            print(f"  [OK] {filename}: {size_mb:.1f} MB - {description}")
        else:
            print(f"  [MISSING] {filename} - {description}")
    
    # Load statistics
    print("\n" + "="*80)
    print("3. DATASET STATISTICS")
    print("="*80)
    
    stats_file = output_dir / "training_dataset_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"\nTraining Dataset:")
        print(f"  Total examples: {stats['total_examples']:,}")
        print(f"  Average text length: {stats['average_text_length']:.0f} characters")
        print(f"\nLabel Distribution:")
        for label, count in sorted(stats['label_distribution'].items()):
            label_name = {0: "severe (depression)", 1: "moderate (anxiety/ptsd)"}.get(int(label), f"unknown({label})")
            print(f"  Label {label} ({label_name}): {count:,}")
        print(f"\nSplits:")
        print(f"  Train: {stats['split_sizes']['train']:,} ({stats['split_ratios']['train']*100:.0f}%)")
        print(f"  Val: {stats['split_sizes']['val']:,} ({stats['split_ratios']['val']*100:.0f}%)")
        print(f"  Test: {stats['split_sizes']['test']:,} ({stats['split_ratios']['test']*100:.0f}%)")
    
    # Matching report
    match_file = output_dir / "label_matching_report.json"
    if match_file.exists():
        with open(match_file, 'r') as f:
            match_stats = json.load(f)
        print("\n" + "="*80)
        print("4. LABELING STATISTICS")
        print("="*80)
        print(f"\nMatching Results:")
        print(f"  Total users processed: {match_stats['total_users_processed']:,}")
        print(f"  Successfully labeled: {match_stats['users_labeled']:,}")
        print(f"  Unlabeled: {match_stats['users_unlabeled']:,}")
        print(f"  Match rate: {match_stats['match_rate_percent']:.1f}%")
        print(f"\nLabel Distribution (in labeled users):")
        for label, count in sorted(match_stats['label_distribution'].items()):
            label_name = {0: "severe (depression)", 1: "moderate (anxiety/ptsd)"}.get(int(label), f"unknown({label})")
            print(f"  Label {label} ({label_name}): {count:,}")
else:
    print("\n[WARNING] Output directory not found!")

print("\n" + "="*80)
print("5. PIPELINE EXECUTION STATUS")
print("="*80)

# Check if pipeline has been run
pipeline_complete = True
if not (output_dir / "rmh_username_labels.json").exists():
    print("\n[ ] Script 1: extract_rmh_labels.py - NOT RUN")
    pipeline_complete = False
else:
    print("\n[OK] Script 1: extract_rmh_labels.py - COMPLETE")

if not (output_dir / "all_labeled_users.json").exists():
    print("[ ] Script 2: match_labels_to_agg_packet.py - NOT RUN")
    pipeline_complete = False
else:
    print("[OK] Script 2: match_labels_to_agg_packet.py - COMPLETE")

if not (output_dir / "validation_report.json").exists():
    print("[ ] Script 3: validate_labeled_data.py - NOT RUN")
    pipeline_complete = False
else:
    print("[OK] Script 3: validate_labeled_data.py - COMPLETE")

if not (output_dir / "train.json").exists():
    print("[ ] Script 4: prepare_training_dataset.py - NOT RUN")
    pipeline_complete = False
else:
    print("[OK] Script 4: prepare_training_dataset.py - COMPLETE")

print("\n" + "="*80)
print("6. TRAINING READINESS")
print("="*80)

training_ready = (
    (output_dir / "train.json").exists() and
    (output_dir / "val.json").exists() and
    (output_dir / "test.json").exists()
)

if training_ready:
    print("\n[OK] Training data ready!")
    print("  - train.json exists")
    print("  - val.json exists")
    print("  - test.json exists")
    print("\n[READY] You can start model training now!")
else:
    print("\n[ ] Training data not ready")
    print("  Run prepare_training_dataset.py first")

print("\n" + "="*80)
print("7. OVERALL STATUS")
print("="*80)

if pipeline_complete and training_ready:
    print("\n[SUCCESS] All pipeline steps completed!")
    print("[READY] Training dataset prepared and ready")
    print("\nNext step: Start model training")
    print("  Run: python start_training.py")
    print("  Or: python modelB_training.py --dataset-path ... --label-field label --train --auto")
else:
    print("\n[IN PROGRESS] Some steps may need to be completed")

print("\n" + "="*80)
print("8. FILE LOCATIONS")
print("="*80)
print(f"\nScripts: {scripts_dir}")
print(f"Outputs: {output_dir}")
print(f"Training script: {scripts_dir.parent.parent / 'modelB_training.py'}")

print("\n" + "="*80)
print("STATUS REPORT COMPLETE")
print("="*80)

