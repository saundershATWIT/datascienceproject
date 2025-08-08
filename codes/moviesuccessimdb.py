# codes/get_imdb.py
from pathlib import Path
import pandas as pd
import kagglehub

# 1) Download
cache_dir = Path(kagglehub.dataset_download("ashirwadsangwan/imdb-dataset"))
print("KaggleHub cache:", cache_dir)

# 2) Point to TSVs
basics = cache_dir / "title.basics.tsv"
ratings = cache_dir / "title.ratings.tsv"

# 3) Load (note: IMDb uses \N for missing)
read_kw = dict(sep="\t", na_values="\\N", low_memory=False)

df_basics = pd.read_csv(basics, **read_kw)
df_ratings = pd.read_csv(ratings, **read_kw)

# 4) Merge on tconst
df = df_basics.merge(df_ratings, on="tconst", how="left")

# Optional: keep only useful columns
keep = [
    "tconst", "primaryTitle", "originalTitle", "titleType",
    "startYear", "runtimeMinutes", "genres",
    "averageRating", "numVotes",
]
df = df[keep]

# 5) Save to your project data folder
out_path = Path("data/imdb_titles_with_ratings.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print("Saved:", out_path.resolve())

# Quick peek
print(df.head())
