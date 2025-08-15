#!/usr/bin/env python3
"""
Script to repair corrupted Reddit checkpoint JSON files.
"""

import json
import os
from pathlib import Path

def repair_checkpoint_file(file_path):
    """Repair a corrupted checkpoint JSON file."""
    print(f"🔧 Repairing corrupted checkpoint file: {file_path}")
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📊 File size: {len(content):,} characters")
    
    # Find the last complete JSON object
    # Look for the pattern that indicates the end of a user object
    import re
    pattern = r'"last_activity":\s*\d+\.?\d*\s*}'
    matches = list(re.finditer(pattern, content))
    
    if matches:
        last_match = matches[-1]
        last_complete_pos = last_match.end()
        print(f"✅ Found last complete user object at position {last_complete_pos:,}")
    else:
        print("❌ Could not find 'last_activity' pattern")
        return False
    
    # Truncate the content to the last complete object
    repaired_content = content[:last_complete_pos]
    
    # Add the closing bracket for the main array
    repaired_content += "\n]"
    
    # Validate the repaired JSON
    try:
        json.loads(repaired_content)
        print("✅ Repaired JSON is valid")
    except json.JSONDecodeError as e:
        print(f"❌ Repaired JSON is still invalid: {e}")
        return False
    
    # Create backup of original file
    backup_path = str(file_path) + ".backup"
    os.rename(file_path, backup_path)
    print(f"💾 Original file backed up to: {backup_path}")
    
    # Write the repaired content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(repaired_content)
    
    print(f"✅ Repaired file written to: {file_path}")
    print(f"📊 Repaired file size: {len(repaired_content):,} characters")
    
    return True

def main():
    """Main function to repair the corrupted checkpoint file."""
    checkpoint_file = Path("F:/DATA STORAGE/AGPacket/reddit_user_analysis_checkpoint_20250814_175758.json")
    
    if not checkpoint_file.exists():
        print(f"❌ File not found: {checkpoint_file}")
        return
    
    if repair_checkpoint_file(checkpoint_file):
        print("🎉 Checkpoint file repaired successfully!")
    else:
        print("💥 Failed to repair checkpoint file")

if __name__ == "__main__":
    main()
