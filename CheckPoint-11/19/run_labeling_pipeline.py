"""
Master Script: Run the complete labeling pipeline.

This script orchestrates all labeling steps in sequence.
"""

import sys
from pathlib import Path
import importlib.util

def run_script(script_name, description):
    """Run a script and handle errors."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False
    
    # Import and run the script
    spec = importlib.util.spec_from_file_location(script_name.replace('.py', ''), script_path)
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
        if hasattr(module, 'main'):
            module.main()
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to run {script_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the complete labeling pipeline."""
    print("="*80)
    print("LABELING PIPELINE - Complete Execution")
    print("="*80)
    print("\nThis will run all labeling scripts in sequence:")
    print("  1. Extract RMH labels")
    print("  2. Match labels to AGG_PACKET")
    print("  3. Validate labeled data")
    print("  4. Prepare training dataset")
    print("\n" + "="*80)
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    steps = [
        ("extract_rmh_labels.py", "Extract RMH Labels"),
        ("match_labels_to_agg_packet.py", "Match Labels to AGG_PACKET"),
        ("validate_labeled_data.py", "Validate Labeled Data"),
        ("prepare_training_dataset.py", "Prepare Training Dataset")
    ]
    
    for script_name, description in steps:
        success = run_script(script_name, description)
        
        if not success:
            print(f"\n{'='*80}")
            print(f"PIPELINE FAILED at: {description}")
            print(f"{'='*80}")
            print("\nYou can:")
            print("  1. Fix the error and re-run this script")
            print("  2. Run individual scripts manually")
            sys.exit(1)
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*80}")
    print("\nAll steps completed successfully!")
    print("Your labeled training dataset is ready in the output directory.")
    print("\nNext steps:")
    print("  - Check the validation report for data quality")
    print("  - Review training_dataset_stats.json for dataset info")
    print("  - Use train.json, val.json, test.json for training")

if __name__ == "__main__":
    main()

