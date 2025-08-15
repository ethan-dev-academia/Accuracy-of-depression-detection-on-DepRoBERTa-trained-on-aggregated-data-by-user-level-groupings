#!/usr/bin/env python3
"""
Processed Files Integrity Checker
Checks all files in the processed folder for corruption, format issues, and data integrity.
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import hashlib

class ProcessedFilesChecker:
    def __init__(self, processed_dir="F:/DATA STORAGE/AGPacket"):
        """
        Initialize the file checker.
        
        Args:
            processed_dir (str): Path to the processed files directory
        """
        self.processed_dir = Path(processed_dir)
        self.results = {
            'total_files': 0,
            'valid_files': 0,
            'corrupted_files': 0,
            'format_issues': 0,
            'encoding_issues': 0,
            'file_details': []
        }
    
    def check_file_integrity(self, file_path):
        """
        Check a single file for various types of corruption and issues.
        
        Args:
            file_path (Path): Path to the file to check
            
        Returns:
            dict: File check results
        """
        file_info = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime),
            'status': 'unknown',
            'issues': [],
            'user_count': 0,
            'file_type': 'unknown'
        }
        
        try:
            # Determine file type based on extension and content
            if file_path.suffix.lower() == '.json':
                file_info['file_type'] = 'json'
                result = self._check_json_file(file_path)
            elif file_path.suffix.lower() == '.csv':
                file_info['file_type'] = 'csv'
                result = self._check_csv_file(file_path)
            else:
                file_info['file_type'] = 'other'
                result = self._check_other_file(file_path)
            
            # Update file info with check results
            file_info.update(result)
            
        except Exception as e:
            file_info['status'] = 'error'
            file_info['issues'].append(f"Check failed: {str(e)}")
        
        return file_info
    
    def _check_json_file(self, file_path):
        """Check JSON file for corruption and format issues."""
        result = {
            'status': 'unknown',
            'issues': [],
            'user_count': 0,
            'structure_valid': False,
            'encoding_valid': False
        }
        
        try:
            # Check file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                result['status'] = 'corrupted'
                result['issues'].append("File is empty (0 bytes)")
                return result
            
            # Try different encoding approaches
            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None
            used_encoding = None
            
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    used_encoding = encoding
                    result['encoding_valid'] = True
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                result['status'] = 'corrupted'
                result['issues'].append("Cannot decode file with any encoding")
                return result
            
            # Try to parse JSON
            try:
                data = json.loads(content)
                result['structure_valid'] = True
                
                # Analyze JSON structure
                if isinstance(data, list):
                    result['user_count'] = len(data)
                    result['status'] = 'valid'
                    
                    # Check if it's user analysis data
                    if data and isinstance(data[0], dict) and 'username' in data[0]:
                        result['file_type'] = 'user_analysis'
                    else:
                        result['file_type'] = 'generic_list'
                        
                elif isinstance(data, dict):
                    result['status'] = 'valid'
                    
                    # Check if it's content samples data
                    if 'users' in data and 'metadata' in data:
                        result['file_type'] = 'content_samples'
                        result['user_count'] = len(data['users']) if 'users' in data else 0
                    elif 'username' in data:
                        result['file_type'] = 'single_user'
                        result['user_count'] = 1
                    else:
                        result['file_type'] = 'generic_dict'
                else:
                    result['status'] = 'format_issue'
                    result['issues'].append("JSON structure is not list or dict")
                
            except json.JSONDecodeError as e:
                result['status'] = 'corrupted'
                result['issues'].append(f"JSON parse error: {str(e)}")
                
                # Try to find the line with the error
                try:
                    lines = content.split('\n')
                    if e.lineno < len(lines):
                        problematic_line = lines[e.lineno - 1]
                        result['issues'].append(f"Problematic line {e.lineno}: {problematic_line[:100]}...")
                except:
                    pass
        
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"Unexpected error: {str(e)}")
        
        return result
    
    def _check_csv_file(self, file_path):
        """Check CSV file for corruption and format issues."""
        result = {
            'status': 'unknown',
            'issues': [],
            'user_count': 0,
            'structure_valid': False,
            'encoding_valid': False
        }
        
        try:
            # Check file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                result['status'] = 'corrupted'
                result['issues'].append("File is empty (0 bytes)")
                return result
            
            # Try to read CSV with pandas
            try:
                df = pd.read_csv(file_path)
                result['structure_valid'] = True
                result['user_count'] = len(df)
                result['status'] = 'valid'
                
                # Check for expected columns
                if len(df.columns) >= 2:
                    result['issues'].append(f"Has {len(df.columns)} columns (expected at least 2)")
                else:
                    result['status'] = 'format_issue'
                    result['issues'].append(f"Only {len(df.columns)} columns (need at least 2)")
                
            except Exception as e:
                result['status'] = 'corrupted'
                result['issues'].append(f"CSV read error: {str(e)}")
        
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"Unexpected error: {str(e)}")
        
        return result
    
    def _check_other_file(self, file_path):
        """Check other file types."""
        result = {
            'status': 'unknown',
            'issues': [],
            'user_count': 0,
            'structure_valid': False,
            'encoding_valid': False
        }
        
        try:
            # Check if file is readable
            with open(file_path, 'rb') as f:
                content = f.read(1024)  # Read first 1KB
            
            if len(content) == 0:
                result['status'] = 'corrupted'
                result['issues'].append("File is empty")
            else:
                result['status'] = 'valid'
                result['issues'].append("File is readable but type not analyzed")
        
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"File access error: {str(e)}")
        
        return result
    
    def calculate_file_hash(self, file_path):
        """Calculate MD5 hash of file for integrity checking."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            return f"Hash failed: {str(e)}"
    
    def check_all_files(self):
        """Check all files in the processed directory."""
        print("🔍 Processed Files Integrity Checker")
        print("=" * 60)
        print(f"Checking directory: {self.processed_dir}")
        print()
        
        if not self.processed_dir.exists():
            print(f"❌ Directory not found: {self.processed_dir}")
            return
        
        # Get all files
        all_files = []
        for file_path in self.processed_dir.rglob("*"):
            if file_path.is_file():
                all_files.append(file_path)
        
        all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"Found {len(all_files)} files to check...")
        print()
        
        # Check each file
        for i, file_path in enumerate(all_files, 1):
            print(f"[{i}/{len(all_files)}] Checking: {file_path.name}")
            
            file_info = self.check_file_integrity(file_path)
            
            # Add hash for large files
            if file_info['file_size_mb'] > 10:  # Files larger than 10MB
                print(f"  📊 Calculating hash for large file...")
                file_info['file_hash'] = self.calculate_file_hash(file_path)
            
            self.results['file_details'].append(file_info)
            self.results['total_files'] += 1
            
            # Update counters
            if file_info['status'] == 'valid':
                self.results['valid_files'] += 1
            elif file_info['status'] == 'corrupted':
                self.results['corrupted_files'] += 1
            elif file_info['status'] == 'format_issue':
                self.results['format_issues'] += 1
            elif file_info['status'] == 'error':
                self.results['encoding_issues'] += 1
            
            # Print status
            status_emoji = {
                'valid': '✅',
                'corrupted': '❌',
                'format_issue': '⚠️',
                'error': '💥',
                'unknown': '❓'
            }
            
            print(f"  {status_emoji.get(file_info['status'], '❓')} {file_info['status'].upper()}")
            
            if file_info['issues']:
                for issue in file_info['issues']:
                    print(f"    • {issue}")
            
            if file_info['user_count'] > 0:
                print(f"    👥 Users: {file_info['user_count']:,}")
            
            print(f"    📁 Size: {file_info['file_size_mb']:.2f} MB")
            print()
        
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of all checks."""
        print("=" * 60)
        print("📊 INTEGRITY CHECK SUMMARY")
        print("=" * 60)
        
        print(f"Total files checked: {self.results['total_files']}")
        print(f"✅ Valid files: {self.results['valid_files']}")
        print(f"❌ Corrupted files: {self.results['corrupted_files']}")
        print(f"⚠️  Format issues: {self.results['format_issues']}")
        print(f"💥 Encoding issues: {self.results['encoding_issues']}")
        
        if self.results['corrupted_files'] > 0 or self.results['format_issues'] > 0:
            print(f"\n🚨 ISSUES FOUND:")
            
            for file_info in self.results['file_details']:
                if file_info['status'] in ['corrupted', 'format_issue']:
                    print(f"  • {file_info['file_name']}: {file_info['status']}")
                    for issue in file_info['issues']:
                        print(f"    - {issue}")
        
        # File type breakdown
        file_types = {}
        for file_info in self.results['file_details']:
            file_type = file_info.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        print(f"\n📁 File Type Breakdown:")
        for file_type, count in sorted(file_types.items()):
            print(f"  {file_type}: {count}")
        
        # Total user count
        total_users = sum(file_info.get('user_count', 0) for file_info in self.results['file_details'])
        print(f"\n👥 Total users across all files: {total_users:,}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"F:/DATA STORAGE/AGPacket/integrity_check_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {str(e)}")
    
    def fix_corrupted_files(self, auto_fix=False):
        """Attempt to fix corrupted files."""
        print(f"\n🔧 FILE REPAIR MODE")
        print("=" * 60)
        
        corrupted_files = [f for f in self.results['file_details'] if f['status'] == 'corrupted']
        
        if not corrupted_files:
            print("✅ No corrupted files found!")
            return
        
        print(f"Found {len(corrupted_files)} corrupted files:")
        for file_info in corrupted_files:
            print(f"  • {file_info['file_name']}")
        
        if not auto_fix:
            response = input(f"\nDo you want to attempt to fix these files? (y/n): ").lower()
            if response != 'y':
                print("File repair cancelled.")
                return
        
        for file_info in corrupted_files:
            print(f"\n🔧 Attempting to fix: {file_info['file_name']}")
            
            if file_info['file_type'] == 'json':
                success = self._fix_json_file(file_info)
            else:
                print(f"  ⚠️  Cannot auto-fix {file_info['file_type']} files")
                success = False
            
            if success:
                print(f"  ✅ Successfully fixed: {file_info['file_name']}")
            else:
                print(f"  ❌ Failed to fix: {file_info['file_name']}")
    
    def _fix_json_file(self, file_info):
        """Attempt to fix a corrupted JSON file."""
        try:
            input_file = file_info['file_path']
            output_file = input_file.replace('.json', '_FIXED.json')
            
            # Read file in binary mode
            with open(input_file, 'rb') as f:
                binary_content = f.read()
            
            # Try to decode with error handling
            try:
                content = binary_content.decode('utf-8', errors='replace')
            except:
                content = binary_content.decode('latin-1')
            
            # Try to parse JSON
            try:
                data = json.loads(content)
                
                # Write fixed file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                return True
                
            except json.JSONDecodeError:
                return False
        
        except Exception:
            return False

def main():
    """Main function to run the integrity checker."""
    
    # Parse command line arguments
    auto_fix = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "--auto-fix":
            auto_fix = True
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage: python check_processed_files.py [OPTIONS]")
            print("\nOptions:")
            print("  --auto-fix    Automatically attempt to fix corrupted files")
            print("  --help, -h    Show this help message")
            return
    
    # Initialize checker
    checker = ProcessedFilesChecker()
    
    # Check all files
    checker.check_all_files()
    
    # Offer to fix corrupted files
    if checker.results['corrupted_files'] > 0:
        checker.fix_corrupted_files(auto_fix=auto_fix)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Check interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
