
from pathlib import Path
import pandas as pd

BASE = Path("data/processed/aluva")

print("AVAILABLE DEMAND FILES")
print("-----------------------------")

files = sorted(BASE.rglob("*demand*.csv"))

if not files:
    print("No demand CSV files found.")
else:
    for file in files:
        print(f"\nFile: {file}")
        df = pd.read_csv(file)

        print("Rows:", len(df))
        print("Columns:", list(df.columns))
        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

print("\nSUCCESS! Demand file inspection completed.")
