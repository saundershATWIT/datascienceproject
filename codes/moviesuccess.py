# pip install kagglehub[pandas-datasets] pandas

from kagglehub import KaggleDatasetAdapter
import kagglehub
from pathlib import Path
import pandas as pd

# 1) Download latest version (also returns the cache path)
path = kagglehub.dataset_download("therealsampat/predict-movie-success-rate")
print("Path to dataset files:", path)

# 2) Try loading by explicit filename first
FILE_NAME = "Movie_classification.csv"  # <- this is the CSV in that dataset

try:
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "therealsampat/predict-movie-success-rate",
        FILE_NAME,
        pandas_kwargs={"low_memory": False},
    )
except Exception as e:
    # 3) Fallback: auto-detect first CSV in the downloaded folder
    print("Direct load failed, falling back to auto-detect. Error:", e)
    root = Path(path)
    csvs = list(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {root}")
    print("Auto-detected CSV:", csvs[0].name)
    df = pd.read_csv(csvs[0])

print("First 5 records:\n", df.head())
