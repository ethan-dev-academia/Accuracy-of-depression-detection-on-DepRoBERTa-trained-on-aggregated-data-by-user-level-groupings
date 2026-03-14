import pandas as pd
from pathlib import Path

rmh_dir = Path(r'F:/DATA STORAGE/RMH Dataset')
csv_files = list(rmh_dir.rglob('*.csv'))[:5]  # Check first 5 files

for csv_file in csv_files:
    print(f'\n{"="*60}')
    print(f'File: {csv_file.name}')
    print(f'{"="*60}')
    try:
        df = pd.read_csv(csv_file, nrows=5)
        print(f'Columns: {list(df.columns)}')
        print(f'Shape: {df.shape}')
        print('\nFirst 3 rows:')
        print(df.head(3).to_string())
    except Exception as e:
        print(f'Error: {e}')

