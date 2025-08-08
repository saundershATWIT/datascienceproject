# pip install kagglehub pandas

from pathlib import Path
import sys
import pandas as pd
import kagglehub

HANDLE = "therealsampat/predict-movie-success-rate"   # your link
FALLBACK = "rounakbanik/the-movies-dataset"           # stable public alt

def load_first_table_from(handle: str) -> pd.DataFrame:
    path = Path(kagglehub.dataset_download(handle))   # may raise 404
    print(f"\nDownloaded to: {path}\nFiles:")
    for f in sorted(path.rglob("*")):
        if f.is_file():
            print(" -", f.relative_to(path))

    # Prefer CSV, then TSV
    csvs = list(path.rglob("*.csv"))
    if csvs:
        print("\nLoading:", csvs[0].name)
        return pd.read_csv(csvs[0], low_memory=False)
    tsvs = list(path.rglob("*.tsv"))
    if tsvs:
        print("\nLoading:", tsvs[0].name)
        return pd.read_csv(tsvs[0], sep="\t", na_values="\\N", low_memory=False)

    raise FileNotFoundError("No CSV/TSV files found in this dataset.")

try:
    df = load_first_table_from(HANDLE)
except Exception as e:
    print(f"\nPrimary dataset failed: {e}\nFalling back to: {FALLBACK}")
    df = load_first_table_from(FALLBACK)

print("\nPreview:")
print(df.head())
