"""Stop any background training processes"""
import subprocess
import sys

print("Checking for running Python processes...")
try:
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                          capture_output=True, text=True, shell=True)
    if 'python.exe' in result.stdout:
        print("Found Python processes:")
        lines = [l for l in result.stdout.split('\n') if 'python.exe' in l and 'PID' not in l]
        for line in lines:
            print(f"  {line.strip()}")
        print("\nNote: You may want to manually stop these if they're training processes.")
        print("You can use Task Manager or: taskkill /PID <pid> /F")
    else:
        print("No Python processes found running.")
except Exception as e:
    print(f"Could not check processes: {e}")

print("\n" + "="*80)
print("Starting training in FOREGROUND mode now...")
print("="*80)

