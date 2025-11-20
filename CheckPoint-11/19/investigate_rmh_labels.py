"""
Deep investigation to find where depression labels might be stored.
"""
import pandas as pd
from pathlib import Path
from collections import Counter
import json

rmh_dir = Path(r'F:/DATA STORAGE/RMH Dataset')

print("="*80)
print("INVESTIGATING RMH DATASET FOR DEPRESSION LABELS")
print("="*80)

# 1. Check all file types
print("\n1. FILE STRUCTURE:")
print("-"*80)
all_files = list(rmh_dir.rglob('*'))
csv_files = [f for f in all_files if f.suffix.lower() == '.csv']
json_files = [f for f in all_files if f.suffix.lower() == '.json']
txt_files = [f for f in all_files if f.suffix.lower() == '.txt']
xlsx_files = [f for f in all_files if f.suffix.lower() == '.xlsx']
other_files = [f for f in all_files if f.suffix and f.suffix.lower() not in ['.csv', '.json', '.txt', '.xlsx']]

print(f"Total files: {len(all_files)}")
print(f"CSV files: {len(csv_files)}")
print(f"JSON files: {len(json_files)}")
print(f"TXT files: {len(txt_files)}")
print(f"XLSX files: {len(xlsx_files)}")
print(f"Other files: {len(other_files)}")

# 2. Look for files with label-related keywords
print("\n2. FILES WITH LABEL-RELATED KEYWORDS:")
print("-"*80)
keywords = ['label', 'depression', 'target', 'severity', 'class', 'annotat', 'gold', 'truth']
for keyword in keywords:
    matches = [f.name for f in all_files if keyword.lower() in f.name.lower()]
    if matches:
        print(f"\n'{keyword}':")
        for match in matches[:5]:
            print(f"  - {match}")

# 3. Check README or documentation
print("\n3. DOCUMENTATION FILES:")
print("-"*80)
doc_files = [f for f in all_files if 'readme' in f.name.lower() or 'doc' in f.name.lower() or 'info' in f.name.lower()]
for doc in doc_files:
    print(f"  - {doc.relative_to(rmh_dir)}")
    try:
        with open(doc, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(1000)  # First 1000 chars
            print(f"    Preview: {content[:200]}...")
    except:
        pass

# 4. Analyze CSV structure in detail
print("\n4. DETAILED CSV ANALYSIS:")
print("-"*80)

# Sample different CSV files
sample_files = csv_files[:10]
for csv_file in sample_files:
    print(f"\nFile: {csv_file.name}")
    try:
        df = pd.read_csv(csv_file, nrows=50)
        cols = list(df.columns)
        
        # Check for author/username columns
        username_cols = [c for c in cols if any(term in c.lower() for term in ['author', 'user', 'name', 'username', 'id'])]
        if username_cols:
            print(f"  Username columns: {username_cols}")
            print(f"  Sample usernames: {df[username_cols[0]].head(3).tolist()}")
        
        # Check for subreddit column
        if 'subreddit' in cols:
            subreddits = df['subreddit'].unique()
            print(f"  Subreddits found: {list(subreddits)}")
            print(f"  Subreddit counts: {dict(Counter(df['subreddit']))}")
        
        # Check ALL columns for potential labels
        potential_label_cols = []
        for col in cols:
            col_lower = col.lower()
            if any(term in col_lower for term in ['label', 'class', 'target', 'severity', 'depression', 'category', 'type', 'status']):
                potential_label_cols.append(col)
        
        if potential_label_cols:
            print(f"  Potential label columns: {potential_label_cols}")
            for col in potential_label_cols:
                unique_vals = df[col].unique()
                print(f"    {col}: {list(unique_vals[:10])}")
        
        # Check numeric columns that might be labels (0, 1, 2)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols[:5]:  # Check first 5 numeric columns
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 5 and all(v in [0, 1, 2, 3, 4] for v in unique_vals):
                print(f"  Possible label column (numeric): {col} - values: {sorted(unique_vals)}")
        
        # Check for binary/ternary categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 5:
                vals_str = ' '.join([str(v)[:20] for v in unique_vals[:5]])
                if any(term in vals_str.lower() for term in ['severe', 'moderate', 'depression', 'normal', 'control']):
                    print(f"  Possible label column (text): {col}")
                    print(f"    Values: {list(unique_vals)}")
        
    except Exception as e:
        print(f"  Error reading file: {e}")

# 5. Check if subreddits map to labels
print("\n5. SUBREDDIT TO LABEL MAPPING ANALYSIS:")
print("-"*80)
subreddit_to_label_map = {
    'depression': 0,  # severe
    'depression_help': 0,
    'depressed': 0,
    'suicidewatch': 0,
    'anxiety': 1,  # moderate
    'anxietyhelp': 1,
    'anxious': 1,
    'ptsd': 1,
    'adhd': None,  # unclear
    'addiction': None,
    'mentalhealth': None,
    'selfhelp': None,
}

# Check a larger sample
for csv_file in csv_files[:5]:
    try:
        df = pd.read_csv(csv_file, nrows=1000)
        if 'subreddit' in df.columns and 'author' in df.columns:
            print(f"\n{csv_file.name}:")
            subreddit_dist = Counter(df['subreddit'])
            print(f"  Subreddit distribution: {dict(subreddit_dist)}")
            
            # Check if subreddit names suggest labels
            for subreddit in subreddit_dist.keys():
                sub_lower = subreddit.lower()
                if 'depress' in sub_lower or 'suicid' in sub_lower:
                    print(f"    '{subreddit}' might indicate SEVERE (0)")
                elif 'anxiet' in sub_lower or 'ptsd' in sub_lower:
                    print(f"    '{subreddit}' might indicate MODERATE (1)")
                elif 'control' in sub_lower or 'normal' in sub_lower or 'happy' in sub_lower:
                    print(f"    '{subreddit}' might indicate NOT DEPRESSION (2)")
    except:
        pass

# 6. Check JSON files if any
print("\n6. JSON FILES ANALYSIS:")
print("-"*80)
for json_file in json_files[:5]:
    print(f"\n{json_file.name}")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            print(f"  Type: dict with keys: {list(data.keys())[:10]}")
            # Check for label keys
            for key in data.keys():
                if any(term in key.lower() for term in ['label', 'target', 'depression', 'class']):
                    print(f"    Found potential label key: {key}")
        elif isinstance(data, list):
            print(f"  Type: list with {len(data)} items")
            if len(data) > 0 and isinstance(data[0], dict):
                print(f"  First item keys: {list(data[0].keys())[:10]}")
                # Check for label keys
                for key in data[0].keys():
                    if any(term in key.lower() for term in ['label', 'target', 'depression', 'class']):
                        print(f"    Found potential label key: {key}")
    except Exception as e:
        print(f"  Error: {e}")

# 7. Check directory structure for hints
print("\n7. DIRECTORY STRUCTURE:")
print("-"*80)
dirs = sorted([d for d in rmh_dir.iterdir() if d.is_dir()])
print(f"Top-level directories: {[d.name for d in dirs[:10]]}")

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)

