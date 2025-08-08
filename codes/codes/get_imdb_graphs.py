# codes/get_imdb_graphs.py
# pip install pandas matplotlib kagglehub

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

# 1) Download IMDb dataset
cache_dir = Path(kagglehub.dataset_download("ashirwadsangwan/imdb-dataset"))
print("KaggleHub cache:", cache_dir)

# 2) TSV file paths
basics = cache_dir / "title.basics.tsv"
ratings = cache_dir / "title.ratings.tsv"

# 3) Load (IMDb uses \N for missing values)
read_kw = dict(sep="\t", na_values="\\N", low_memory=False)
df_basics = pd.read_csv(basics, **read_kw)
df_ratings = pd.read_csv(ratings, **read_kw)

# 4) Merge on tconst
df = df_basics.merge(df_ratings, on="tconst", how="left")

# 5) Keep only useful columns
keep = [
    "tconst", "primaryTitle", "originalTitle", "titleType",
    "startYear", "runtimeMinutes", "genres",
    "averageRating", "numVotes",
]
df = df[keep].copy()

# Convert numeric columns
for col in ["startYear", "runtimeMinutes", "averageRating", "numVotes"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 6) Save to data folder
out_path = Path("data/imdb_titles_with_ratings.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print("Saved dataset to:", out_path.resolve())

# Make sure pictures folder exists
Path("pictures").mkdir(exist_ok=True)

# --- Graph 1: Distribution of IMDb ratings ---
plt.figure(figsize=(8, 5))
df["averageRating"].dropna().plot(kind="hist", bins=30, color="skyblue")
plt.xlabel("IMDb Rating")
plt.ylabel("Count")
plt.title("Distribution of IMDb Ratings")
plt.tight_layout()
plt.savefig("pictures/imdb_rating_hist.png", dpi=200)
plt.close()
print("Saved pictures/imdb_rating_hist.png")

# --- Graph 2: Number of titles per genre (top 15) ---
g = df.dropna(subset=["genres"]).copy()
g["genres"] = g["genres"].str.split(",")
g = g.explode("genres")
genre_counts = g["genres"].value_counts().nlargest(15)

plt.figure(figsize=(10, 6))
genre_counts.plot(kind="bar", color="salmon")
plt.ylabel("Number of Titles")
plt.xlabel("Genre")
plt.title("Top 15 Genres by Number of Titles")
plt.tight_layout()
plt.savefig("pictures/imdb_genre_counts.png", dpi=200)
plt.close()
print("Saved pictures/imdb_genre_counts.png")

# --- Graph 3: Average rating by genre (top 15) ---
avg_rating_by_genre = g.groupby("genres")["averageRating"].mean().loc[genre_counts.index]

plt.figure(figsize=(10, 6))
avg_rating_by_genre.sort_values().plot(kind="barh", color="green")
plt.xlabel("Average Rating")
plt.ylabel("Genre")
plt.title("Average Rating by Genre (Top 15)")
plt.tight_layout()
plt.savefig("pictures/imdb_avg_rating_by_genre.png", dpi=200)
plt.close()
print("Saved pictures/imdb_avg_rating_by_genre.png")

# --- Graph 4: Runtime vs. average rating scatter ---
plt.figure(figsize=(8, 6))
plt.scatter(df["runtimeMinutes"], df["averageRating"], alpha=0.3)
plt.xlabel("Runtime (Minutes)")
plt.ylabel("IMDb Rating")
plt.title("Runtime vs IMDb Rating")
plt.tight_layout()
plt.savefig("pictures/imdb_runtime_vs_rating.png", dpi=200)
plt.close()
print("Saved pictures/imdb_runtime_vs_rating.png")

print("All graphs saved in 'pictures/' folder.")
