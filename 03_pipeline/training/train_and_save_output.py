"""
Alternative: Run training and save output using subprocess redirection.
This captures everything including progress bars.
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

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
print("\nAll output will be saved to the log file.")
print("Progress will also be displayed in terminal in real-time.")
print("\n" + "="*80 + "\n")

# Open log file
with open(log_file, 'w', encoding='utf-8', buffering=1) as log:
    # Write header
    log.write("="*80 + "\n")
    log.write("TRAINING OUTPUT LOG\n")
    log.write("="*80 + "\n")
    log.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write(f"Training script: {training_script}\n")
    log.write("="*80 + "\n\n")
    
    # Run training script and capture output
    try:
        process = subprocess.Popen(
            [sys.executable, str(training_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output to both file and console
        for line in process.stdout:
            # Write to file
            log.write(line)
            log.flush()
            # Also print to console
            print(line, end='', flush=True)
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            log.write(f"\n[ERROR] Process exited with code {return_code}\n")
            print(f"\n[ERROR] Process exited with code {return_code}")
        
    except KeyboardInterrupt:
        log.write("\n\n[INTERRUPTED] Training stopped by user\n")
        print("\n\n[INTERRUPTED] Training stopped by user")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        log.write(f"\n\n[ERROR] Failed to run training: {e}\n")
        print(f"\n\n[ERROR] Failed to run training: {e}")
        raise
    finally:
        # Write footer
        log.write("\n" + "="*80 + "\n")
        log.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("="*80 + "\n")

print(f"\n" + "="*80)
print(f"Training output saved to: {log_file}")
print("="*80)




