#!/usr/bin/env python3
"""
Enhanced Script to check for duplicate usernames in the second column of CSV datasets.
Completely rewritten with better file detection and more accurate duplicate checking.
Now includes detailed duplicate analysis showing messages and content.
"""

import pandas as pd
import os
from collections import Counter, defaultdict
import sys
from pathlib import Path

def get_all_csv_files(data_dir):
    """
    Get all CSV files from the data directory using pathlib for better reliability.
    
    Args:
        data_dir (str): Path to the data directory
        
    Returns:
        list: List of Path objects for CSV files
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' not found!")
        return []
    
    if not data_path.is_dir():
        print(f"Error: '{data_dir}' is not a directory!")
        return []
    
    # Find all CSV files recursively
    csv_files = list(data_path.rglob("*.csv"))
    
    # Convert to absolute paths and sort
    csv_files = [f.resolve() for f in csv_files]
    csv_files.sort()
    
    return csv_files

def analyze_csv_file(file_path):
    """
    Analyze a single CSV file for duplicate usernames in the second column.
    
    Args:
        file_path (Path): Path to the CSV file
        
    Returns:
        dict: Analysis results or None if error
    """
    try:
        print(f"\nAnalyzing: {file_path.name}")
        print("-" * 50)
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Basic file info
        total_rows = len(df)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Total rows: {total_rows:,}")
        print(f"Columns: {len(df.columns)}")
        
        # Check if we have enough columns
        if len(df.columns) < 2:
            print("❌ Error: File has fewer than 2 columns")
            return None
        
        # Get second column info
        second_col_name = df.columns[1]
        second_col_data = df[second_col_name]
        
        print(f"Second column: '{second_col_name}'")
        
        # Remove NaN values and convert to string
        valid_usernames = second_col_data.dropna().astype(str)
        valid_count = len(valid_usernames)
        
        if valid_count == 0:
            print("❌ Error: No valid usernames found in second column")
            return None
        
        print(f"Valid usernames: {valid_count:,}")
        
        # Count occurrences
        username_counts = Counter(valid_usernames)
        unique_count = len(username_counts)
        
        # Find duplicates (usernames appearing more than once)
        duplicates = {username: count for username, count in username_counts.items() if count > 1}
        duplicate_count = len(duplicates)
        
        print(f"Unique usernames: {unique_count:,}")
        print(f"Duplicate usernames: {duplicate_count}")
        
        # Detailed duplicate analysis
        duplicate_details = {}
        if duplicate_count > 0:
            print("\n🔍 Duplicate usernames found:")
            for username, count in sorted(duplicates.items()):
                print(f"  '{username}': appears {count} times")
                
                # Get all rows where this username appears
                duplicate_rows = df[df[second_col_name] == username]
                duplicate_details[username] = {
                    'count': count,
                    'rows': duplicate_rows
                }
        else:
            print("✅ No duplicate usernames found!")
        
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'second_column': second_col_name,
            'total_rows': total_rows,
            'valid_usernames': valid_count,
            'unique_usernames': unique_count,
            'duplicate_count': duplicate_count,
            'duplicates': duplicates,
            'duplicate_details': duplicate_details,
            'file_size_mb': file_size_mb,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error processing {file_path.name}: {str(e)}")
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'error': str(e),
            'success': False
        }

def show_detailed_duplicates(result):
    """
    Show detailed information about duplicates including messages and content.
    
    Args:
        result (dict): Analysis result containing duplicate details
    """
    if not result['duplicate_details']:
        return
    
    print(f"\n📋 DETAILED DUPLICATE ANALYSIS for {result['file_name']}")
    print("=" * 80)
    
    for username, details in result['duplicate_details'].items():
        print(f"\n👤 Username: '{username}' (appears {details['count']} times)")
        print("-" * 60)
        
        # Get the rows for this duplicate username
        rows = details['rows']
        
        # Show column names for reference
        columns = rows.columns.tolist()
        print(f"Available columns: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}")
        
        # Show each duplicate entry
        for idx, (_, row) in enumerate(rows.iterrows(), 1):
            print(f"\n  Entry {idx}:")
            
            # Show key information - try to find relevant columns
            if 'date' in columns:
                date_val = row['date'] if pd.notna(row['date']) else 'N/A'
                print(f"    Date: {date_val}")
            
            if 'post' in columns:
                post_val = row['post'] if pd.notna(row['post']) else 'N/A'
                # Truncate long posts
                if len(str(post_val)) > 200:
                    post_val = str(post_val)[:200] + "..."
                print(f"    Post: {post_val}")
            
            # Show other potentially interesting columns
            interesting_cols = ['subreddit', 'title', 'content', 'message', 'text', 'body']
            for col in interesting_cols:
                if col in columns and pd.notna(row[col]):
                    val = row[col]
                    if len(str(val)) > 100:
                        val = str(val)[:100] + "..."
                    print(f"    {col.capitalize()}: {val}")
            
            # Show row index for reference
            print(f"    Row index: {idx-1}")

def main():
    """Main function to analyze all CSV files for duplicate usernames."""
    
    print("🔍 CSV Duplicate Username Checker")
    print("=" * 60)
    
    # Get all CSV files
    csv_files = get_all_csv_files("F:/DATA STORAGE/RMH Dataset")
    
    if not csv_files:
        print("No CSV files found in F:/DATA STORAGE/RMH Dataset directory.")
        print("Please place your CSV files in the 'F:/DATA STORAGE/RMH Dataset' folder.")
        return
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for i, file_path in enumerate(csv_files, 1):
        print(f"  {i}. {file_path.name}")
    
    print("\n" + "=" * 60)
    
    # Analyze each file
    results = []
    successful_analyses = 0
    total_duplicates = 0
    total_rows = 0
    total_unique = 0
    
    for i, file_path in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing...")
        
        result = analyze_csv_file(file_path)
        results.append(result)
        
        if result['success']:
            successful_analyses += 1
            total_duplicates += result['duplicate_count']
            total_rows += result['total_rows']
            total_unique += result['unique_usernames']
    
    # Summary report
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    print(f"Files processed: {len(csv_files)}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {len(csv_files) - successful_analyses}")
    
    if successful_analyses > 0:
        print(f"\n📈 Data Statistics:")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Total unique usernames: {total_unique:,}")
        print(f"  Total duplicate usernames: {total_duplicates}")
        
        # Files with duplicates
        files_with_duplicates = [r for r in results if r['success'] and r['duplicate_count'] > 0]
        
        if files_with_duplicates:
            print(f"\n⚠️  Files with duplicate usernames ({len(files_with_duplicates)}):")
            for result in files_with_duplicates:
                print(f"  • {result['file_name']}: {result['duplicate_count']} duplicates")
            
            # Show detailed duplicate information
            print(f"\n" + "=" * 80)
            print("🔍 DETAILED DUPLICATE ANALYSIS")
            print("=" * 80)
            
            for result in files_with_duplicates:
                show_detailed_duplicates(result)
        else:
            print("\n✅ No duplicate usernames found in any file!")
        
        # File size summary
        total_size = sum(r['file_size_mb'] for r in results if r['success'])
        avg_size = total_size / successful_analyses if successful_analyses > 0 else 0
        print(f"\n💾 Storage:")
        print(f"  Total size: {total_size:.2f} MB")
        print(f"  Average size: {avg_size:.2f} MB per file")
    
    # Show any errors
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print(f"\n❌ Failed analyses:")
        for result in failed_results:
            print(f"  • {result['file_name']}: {result['error']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Script interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
