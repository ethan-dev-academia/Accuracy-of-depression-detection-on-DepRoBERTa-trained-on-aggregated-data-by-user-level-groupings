"""
Train model and save ALL output to a text file.
This script captures everything (stdout, stderr) to a log file.
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Get the script directory
script_dir = Path(__file__).parent
training_script = script_dir / "train_final_model.py"

# Create log file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = script_dir / f"training_output_{timestamp}.txt"

print("="*80)
print("TRAINING WITH FULL OUTPUT LOGGING")
print("="*80)
print(f"\nTraining script: {training_script}")
print(f"Output log file: {log_file}")
print("\nAll output will be saved to the log file AND displayed in terminal.")
print("You can watch progress in real-time while it's also being saved.")
print("\n" + "="*80 + "\n")

# Open log file for writing
with open(log_file, 'w', encoding='utf-8') as log:
    # Write header to log file
    log.write("="*80 + "\n")
    log.write("TRAINING OUTPUT LOG\n")
    log.write("="*80 + "\n")
    log.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write(f"Training script: {training_script}\n")
    log.write("="*80 + "\n\n")
    log.flush()
    
    # Create a class to tee output to both file and console
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Redirect stdout and stderr to both console and file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        sys.stdout = Tee(original_stdout, log)
        sys.stderr = Tee(original_stderr, log)
        
        # Run the training script
        exec(open(training_script).read())
        
    except KeyboardInterrupt:
        log.write("\n\n[INTERRUPTED] Training stopped by user\n")
        print("\n\n[INTERRUPTED] Training stopped by user")
    except Exception as e:
        log.write(f"\n\n[ERROR] Training failed: {e}\n")
        import traceback
        log.write(traceback.format_exc())
        print(f"\n\n[ERROR] Training failed: {e}")
        raise
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Write footer to log file
        log.write("\n" + "="*80 + "\n")
        log.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("="*80 + "\n")

print(f"\n" + "="*80)
print(f"Training output saved to: {log_file}")
print("="*80)

