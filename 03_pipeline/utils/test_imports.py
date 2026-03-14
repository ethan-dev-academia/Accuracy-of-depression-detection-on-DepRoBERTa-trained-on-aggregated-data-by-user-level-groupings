#!/usr/bin/env python
"""
Test script to verify all required modules for RESULTS_SECTION_COMPLETE.ipynb
"""

print("Testing imports for RESULTS_SECTION_COMPLETE.ipynb...")
print("-" * 60)

try:
    import matplotlib.pyplot as plt
    print(f"[OK] matplotlib imported successfully (version: {plt.matplotlib.__version__})")
except ImportError as e:
    print(f"[ERROR] matplotlib import failed: {e}")
    exit(1)

try:
    import numpy as np
    print(f"[OK] numpy imported successfully (version: {np.__version__})")
except ImportError as e:
    print(f"[ERROR] numpy import failed: {e}")
    exit(1)

try:
    import json
    print("[OK] json imported successfully (built-in)")
except ImportError as e:
    print(f"[ERROR] json import failed: {e}")
    exit(1)

try:
    import os
    print("[OK] os imported successfully (built-in)")
except ImportError as e:
    print(f"[ERROR] os import failed: {e}")
    exit(1)

print("-" * 60)
print("All required modules are working correctly!")
print("\nYou can now run the notebook without errors.")

