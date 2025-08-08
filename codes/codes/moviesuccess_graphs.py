# pip install pandas matplotlib

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your cached Kaggle dataset
csv_path = Path.home() / ".cache/kagglehub/datasets/therealsampat/predict-movie-success-rate/versions/1/movie_success_rate.csv"

# Load dataset
df = pd.read_csv(csv_path, low_memory=False)
print("Loaded dataset with shape:", df.shape)

# Make sure pictures folder exists
Path("pictures").mkdir(exist_ok=True)

# --- Graph 1: Count of movies per genre ---
if "Genre" in df.columns:
    genre_counts = (
        df["Genre"].dropna()
        .apply(lambda g: g.split(",")[0].strip())
        .value_counts()
    )

    plt.figure(figsize=(10, 6))
    genre_counts.plot(kind="bar", color="skyblue")
    plt.ylabel("Number of Movies")
    plt.xlabel("Genre (first listed)")
    plt.title("Number of Movies per Genre")
    plt.tight_layout()
    plt.savefig("pictures/movies_per_genre.png", dpi=200)
    plt.close()
    print("Saved pictures/movies_per_genre.png")

# --- Graph 2: Histogram of Ratings ---
if "Rating" in df.columns:
    plt.figure(figsize=(8, 5))
    df["Rating"].dropna().plot(kind="hist", bins=20, color="salmon")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.title("Distribution of Ratings")
    plt.tight_layout()
    plt.savefig("pictures/rating_hist.png", dpi=200)
    plt.close()
    print("Saved pictures/rating_hist.png")

# --- Graph 3: Revenue vs Rating scatter ---
if "Revenue (Millions)" in df.columns and "Rating" in df.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Revenue (Millions)"], df["Rating"], alpha=0.5)
    plt.xlabel("Revenue (Millions USD)")
    plt.ylabel("Rating")
    plt.title("Revenue vs Rating")
    plt.tight_layout()
    plt.savefig("pictures/revenue_vs_rating.png", dpi=200)
    plt.close()
    print("Saved pictures/revenue_vs_rating.png")

# --- Graph 4: Success vs Rating scatter ---
if "Success" in df.columns and "Rating" in df.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Success"], df["Rating"], alpha=0.5, color="green")
    plt.xlabel("Success")
    plt.ylabel("Rating")
    plt.title("Success vs Rating")
    plt.tight_layout()
    plt.savefig("pictures/success_vs_rating.png", dpi=200)
    plt.close()
    print("Saved pictures/success_vs_rating.png")
