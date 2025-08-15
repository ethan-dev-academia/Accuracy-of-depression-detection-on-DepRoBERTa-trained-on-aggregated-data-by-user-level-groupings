#!/usr/bin/env python3
"""
Enhanced Script to check for duplicate usernames in the second column of CSV datasets.
Completely rewritten with better file detection and more accurate duplicate checking.
Now includes detailed duplicate analysis showing messages and content.
Enhanced with multiprocessing for faster batch processing.
"""

import pandas as pd
import os
from collections import Counter, defaultdict
import sys
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

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

def analyze_csv_files_batch(file_paths, max_workers=None, show_progress=True):
    """
    Analyze multiple CSV files concurrently using multiprocessing.
    
    Args:
        file_paths (list): List of Path objects for CSV files to analyze
        max_workers (int): Maximum number of worker processes (default: CPU count)
        show_progress (bool): Whether to show progress information
        
    Returns:
        list: List of analysis results
    """
    if not file_paths:
        return []
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(file_paths))
    
    if show_progress:
        print(f"\n🚀 Starting batch analysis with {max_workers} worker processes...")
        print(f"Processing {len(file_paths)} files concurrently...")
    
    start_time = time.time()
    results = []
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(analyze_csv_file, file_path): file_path 
                            for file_path in file_paths}
            
            # Process completed tasks
            completed_count = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if show_progress:
                        status = "✅" if result['success'] else "❌"
                        print(f"[{completed_count}/{len(file_paths)}] {status} {file_path.name}")
                        
                except Exception as e:
                    error_result = {
                        'file_path': str(file_path),
                        'file_name': file_path.name,
                        'error': f"Processing error: {str(e)}",
                        'success': False
                    }
                    results.append(error_result)
                    
                    if show_progress:
                        print(f"[{completed_count}/{len(file_paths)}] ❌ {file_path.name} - Error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Error in batch processing: {str(e)}")
        # Fallback to sequential processing
        print("🔄 Falling back to sequential processing...")
        results = []
        for i, file_path in enumerate(file_paths):
            if show_progress:
                print(f"[{i+1}/{len(file_paths)}] Processing {file_path.name}...")
            result = analyze_csv_file(file_path)
            results.append(result)
    
    elapsed_time = time.time() - start_time
    
    if show_progress:
        print(f"\n⏱️  Batch processing completed in {elapsed_time:.2f} seconds")
        print(f"Average time per file: {elapsed_time/len(file_paths):.2f} seconds")
    
    return results

def analyze_csv_files_optimized(file_paths, use_multiprocessing=True, max_workers=None, 
                               batch_size=10, show_progress=True):
    """
    Optimized function that can process files in batches with multiprocessing.
    
    Args:
        file_paths (list): List of Path objects for CSV files to analyze
        use_multiprocessing (bool): Whether to use multiprocessing
        max_workers (int): Maximum number of worker processes
        batch_size (int): Number of files to process in each batch
        show_progress (bool): Whether to show progress information
        
    Returns:
        list: List of analysis results
    """
    if not file_paths:
        return []
    
    if not use_multiprocessing or len(file_paths) <= 1:
        # Sequential processing for small datasets
        if show_progress:
            print(f"\n🔄 Processing {len(file_paths)} files sequentially...")
        
        results = []
        for i, file_path in enumerate(file_paths):
            if show_progress:
                print(f"[{i+1}/{len(file_paths)}] Processing {file_path.name}...")
            result = analyze_csv_file(file_path)
            results.append(result)
        return results
    
    # Multiprocessing with batching for large datasets
    if show_progress:
        print(f"\n🚀 Optimized batch processing with multiprocessing...")
        print(f"Total files: {len(file_paths)}")
        print(f"Batch size: {batch_size}")
        print(f"Max workers: {max_workers or mp.cpu_count()}")
    
    all_results = []
    total_batches = (len(file_paths) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(file_paths))
        batch_files = file_paths[start_idx:end_idx]
        
        if show_progress:
            print(f"\n📦 Processing batch {batch_num + 1}/{total_batches} ({len(batch_files)} files)...")
        
        batch_results = analyze_csv_files_batch(
            batch_files, 
            max_workers=max_workers, 
            show_progress=show_progress
        )
        all_results.extend(batch_results)
    
    return all_results

def example_usage():
    """
    Example function showing how to use the multiprocessing functionality programmatically.
    This can be called from other scripts or used as a reference.
    """
    print("🔍 Example Usage of Multiprocessing CSV Analyzer")
    print("=" * 60)
    
    # Example 1: Basic multiprocessing
    print("\n📚 Example 1: Basic multiprocessing")
    print("results = analyze_csv_files_optimized(csv_files)")
    
    # Example 2: Custom worker count
    print("\n📚 Example 2: Custom worker count")
    print("results = analyze_csv_files_optimized(csv_files, max_workers=4)")
    
    # Example 3: Custom batch size
    print("\n📚 Example 3: Custom batch size")
    print("results = analyze_csv_files_optimized(csv_files, batch_size=20)")
    
    # Example 4: Sequential processing
    print("\n📚 Example 4: Sequential processing")
    print("results = analyze_csv_files_optimized(csv_files, use_multiprocessing=False)")
    
    # Example 5: Full customization
    print("\n📚 Example 5: Full customization")
    print("results = analyze_csv_files_optimized(")
    print("    csv_files,")
    print("    use_multiprocessing=True,")
    print("    max_workers=6,")
    print("    batch_size=15,")
    print("    show_progress=True")
    print(")")
    
    print("\n💡 Performance Tips:")
    print("• Use multiprocessing for datasets with 5+ files")
    print("• Set max_workers to your CPU core count for best performance")
    print("• Adjust batch_size based on available memory")
    print("• For very large datasets, use smaller batch sizes")

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
    
    # Parse command line arguments
    use_multiprocessing = True
    max_workers = None
    batch_size = 10
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "--no-mp" or arg == "-s":
                use_multiprocessing = False
                print("🔄 Sequential processing mode enabled")
            elif arg.startswith("--workers=") or arg.startswith("-w="):
                try:
                    max_workers = int(arg.split("=")[1])
                    print(f"👥 Using {max_workers} worker processes")
                except ValueError:
                    print("⚠️  Invalid worker count, using default")
            elif arg.startswith("--batch-size=") or arg.startswith("-b="):
                try:
                    batch_size = int(arg.split("=")[1])
                    print(f"📦 Batch size set to {batch_size}")
                except ValueError:
                    print("⚠️  Invalid batch size, using default")
            elif arg == "--help" or arg == "-h":
                print("\nUsage: python check_duplicate_usernames.py [OPTIONS]")
                print("\nOptions:")
                print("  --no-mp, -s          Use sequential processing instead of multiprocessing")
                print("  --workers=N, -w=N    Set number of worker processes (default: CPU count)")
                print("  --batch-size=N, -b=N Set batch size for processing (default: 10)")
                print("  --examples, -e       Show example usage")
                print("  --help, -h           Show this help message")
                print("\nExamples:")
                print("  python check_duplicate_usernames.py")
                print("  python check_duplicate_usernames.py --no-mp")
                print("  python check_duplicate_usernames.py --workers=4 --batch-size=20")
                return
            elif arg == "--examples" or arg == "-e":
                example_usage()
                return
    
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
    
    # Use optimized processing
    start_time = time.time()
    results = analyze_csv_files_optimized(
        csv_files,
        use_multiprocessing=use_multiprocessing,
        max_workers=max_workers,
        batch_size=batch_size,
        show_progress=True
    )
    total_processing_time = time.time() - start_time
    
    # Calculate statistics
    successful_analyses = sum(1 for r in results if r['success'])
    total_duplicates = sum(r['duplicate_count'] for r in results if r['success'])
    total_rows = sum(r['total_rows'] for r in results if r['success'])
    total_unique = sum(r['unique_usernames'] for r in results if r['success'])
    
    # Summary report
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    print(f"Files processed: {len(csv_files)}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {len(csv_files) - successful_analyses}")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    
    if use_multiprocessing:
        print(f"Processing mode: Multiprocessing (batch size: {batch_size})")
    else:
        print("Processing mode: Sequential")
    
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
