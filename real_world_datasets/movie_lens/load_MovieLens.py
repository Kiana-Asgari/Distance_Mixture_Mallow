# ===========================================
# MovieLens: download, choose movies, get full rankings
# ===========================================
import argparse
import itertools
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from typing import Union


# -----------------------------
# 1. Download & extract helper
# -----------------------------
def download_movielens(version: str = "ml-25m", dest: Union[Path, str, None] = None) -> Path:
    """
    Download and unzip a MovieLens dataset. Returns the extraction directory.
    Uses caching to avoid re-downloading or re-extracting if data already exists.

    Parameters
    ----------
    version : str
        One of the official archives, e.g. "ml-25m", "ml-1m", "ml-latest-small".
    dest : Path | str | None
        Where to put the extracted files (directory will be created if necessary).
        If None, uses a cache directory in the current file's parent directory.

    Returns
    -------
    Path
        Directory containing ratings.csv (or ratings.dat) and movies.csv (or movies.dat).
    """
    base_url = f"https://files.grouplens.org/datasets/movielens/{version}.zip"
    
    # Use a persistent cache directory instead of temporary directory
    if dest is None:
        cache_dir = Path(__file__).parent / "cache"
        dest_dir = cache_dir
    else:
        dest_dir = Path(dest).expanduser().resolve()
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_dir = dest_dir / version
    zip_path = dest_dir / f"{version}.zip"
    
    # Check if already extracted
    if extracted_dir.exists():
        print(f"  ✔ Using cached data from {extracted_dir}")
        return extracted_dir
    
    # Download zip if needed
    if not zip_path.exists():
        print(f"Downloading {version} … (~please wait)")
        urlretrieve(base_url, zip_path)
        print("  ✔ download finished")
    else:
        print(f"  ✔ Using cached zip file")
    
    # Extract
    print("Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"  ✔ files available in {extracted_dir}")

    return extracted_dir


# -----------------------------------
# 2. Read ratings + movies as DataFrames
# -----------------------------------
def load_ratings_movies(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ratings and movie metadata into pandas DataFrames,
    regardless of whether the dataset is CSV (:: delimited) or DAT (pipe :: delimited).
    """
    # Decide between CSV and ::: DAT layout
    ratings_file_csv = data_dir / "ratings.csv"
    ratings_file_dat = data_dir / "ratings.dat"

    if ratings_file_csv.exists():  # e.g. ml‑25m, ml‑latest-small
        ratings = pd.read_csv(ratings_file_csv, usecols=["userId", "movieId", "rating", "timestamp"])
        movies = pd.read_csv(data_dir / "movies.csv", usecols=["movieId", "title"])
    else:  # older “1M” layout uses ‘::’
        ratings = pd.read_csv(
            ratings_file_dat,
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
            usecols=[0, 1, 2, 3],
        )
        movies = pd.read_csv(
            data_dir / "movies.dat",
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            usecols=[0, 1],
        )

    return ratings, movies


# --------------------------------------------------------------
# 3a. Greedy movie‑set selection maximising common‑user coverage
# --------------------------------------------------------------
def greedy_movie_set(
    ratings: pd.DataFrame,
    candidate_pool_size: int,
    target_set_size: int,
    min_ratings: int,
) -> list[int]:
    """
    Greedy heuristic: pick movies so that the number of users
    who have rated *all* films selected so far is always maximised.

    Returns
    -------
    list[int]  (movieIds)
    """
    # Pre‑filter movies with enough ratings
    counts = (
        ratings.groupby("movieId")
        .size()
        .rename("count")
        .reset_index()
        .query(f"count >= {min_ratings}")
        .sort_values("count", ascending=False)
    )

    candidate_ids = counts.head(candidate_pool_size)["movieId"].tolist()
    if len(candidate_ids) < target_set_size:
        raise ValueError(
            f"Only {len(candidate_ids)} movies satisfy the ≥{min_ratings}‑rating threshold "
            f"(need at least {target_set_size})."
        )

    # Build an index: movieId -> set(users)  (speed boost)
    movie2users = {
        mid: set(ratings.loc[ratings.movieId == mid, "userId"])
        for mid in candidate_ids
    }

    selected: list[int] = []
    common_users: set[int] | None = None

    for step in range(target_set_size):
        best_mid = None
        best_size = -1
        best_users = None

        for mid in candidate_ids:
            if mid in selected:
                continue

            users_this_movie = movie2users[mid]
            candidate_users = (
                users_this_movie
                if common_users is None
                else common_users & users_this_movie
            )
            size = len(candidate_users)

            if size > best_size:
                best_size, best_mid, best_users = size, mid, candidate_users

        if best_mid is None or best_size == 0:
            break  # cannot improve further

        selected.append(best_mid)
        common_users = best_users
        print(
            f"Step {step+1}: picked {best_mid}  —  users with complete set so far: {best_size}"
        )

    return selected


# --------------------------------------------------------
# 3b. Build full rankings for users who rated *all* movies
# --------------------------------------------------------
def build_full_rankings(
    ratings: pd.DataFrame, selected_movie_ids: list[int]
) -> pd.DataFrame:
    """
    Return a tidy DataFrame with columns
      ['userId', 'movieId', 'rating', 'rank']
    where each row belongs to a user who rated *all* selected movies,
    and 'rank' gives the within‑user ordering (1 = favourite).

    Ties (identical ratings) are broken by earlier timestamp.
    """
    # Restrict to our movies
    subset = ratings[ratings.movieId.isin(selected_movie_ids)].copy()

    # Keep only users who rated every movie in the set
    n_needed = len(selected_movie_ids)
    good_users = (
        subset.groupby("userId").movieId.nunique().loc[lambda x: x == n_needed].index
    )
    subset = subset[subset.userId.isin(good_users)]

    # Compute ranks within each user
    subset["rank"] = (
        subset.sort_values(["userId", "rating", "timestamp"], ascending=[True, False, True])
        .groupby("userId")
        .cumcount()
        + 1
    )
    return subset


# -----------------
# 4. Main routine
# -----------------
def main(
    dataset_version: str = "ml-25m",
    top_n: int = 10,
    min_ratings: int = 1_000,
    candidate_pool_size: int = 100,
):
    print(f"▶ Preparing MovieLens {dataset_version}")
    data_dir = download_movielens(dataset_version)
    ratings, movies = load_ratings_movies(data_dir)
    print(f"   • ratings shape: {ratings.shape}")

    # Select movies greedily
    print("\n▶ Selecting movies")
    selected = greedy_movie_set(
        ratings, candidate_pool_size, top_n, min_ratings
    )
    if len(selected) < top_n:
        print(
            f"Warning: only {len(selected)} movies could be found with the given constraints."
        )

    names = movies.set_index("movieId").loc[selected, "title"].tolist()
    print("\nSelected films:")
    for i, (mid, name) in enumerate(zip(selected, names), 1):
        print(f"{i:2d}. {name}  (movieId={mid})")

    # Build full rankings
    print("\n▶ Constructing full‑ranking table …")
    rankings = build_full_rankings(ratings, selected)
    print(
        f"   • {rankings.userId.nunique():,} users provide complete rankings "
        f"over the {len(selected)} movies."
    )

    # Pivot to a compact user‑by‑movie rating matrix (optional):
    rating_matrix = (
        rankings.pivot(index="userId", columns="movieId", values="rating")
        .sort_index()
        .astype(float)
    )
    print("\nSample of the ranking matrix:")
    print(rating_matrix.head())

    return selected, rankings, rating_matrix


# -------------------------------------------------
# If you are running this cell in Colab interactively:
# -------------------------------------------------
def load_and_return_ratings_movies(n_movies: int = 10):
    # The argument parser lets you run the script as
    # !python myscript.py --dataset-version ml-1m --top-n 8
    # when saved to a .py file; but in a notebook it's easier
    # to just call main() with parameters.
    parser = argparse.ArgumentParser(description="MovieLens full‑ranking extractor")
    parser.add_argument("--dataset-version", default="ml-25m")
    parser.add_argument("--min-ratings", type=int, default=1000)
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    args, unknown = parser.parse_known_args()

    selected, rankings, rating_matrix= main(
        dataset_version=args.dataset_version,
        top_n=n_movies,
        min_ratings=args.min_ratings,
        candidate_pool_size=args.candidate_pool_size,
    )
    
    # Convert ratings to rankings format expected by the models
    # Each user's ranking is based on their ratings (higher rating = better rank)
    # Use timestamp as tiebreaker for same ratings
    rankings_list = []
    for user_id in rating_matrix.index:
        user_ratings = rating_matrix.loc[user_id]
        
        # Get user's data with timestamps for tiebreaking
        user_data = rankings[rankings.userId == user_id].copy()
        user_data = user_data.sort_values(['rating', 'timestamp'], ascending=[False, True])
        
        # Convert movie IDs to 1-based indices
        movie_to_index = {movie_id: i+1 for i, movie_id in enumerate(selected)}
        ranking = [movie_to_index[movie_id] for movie_id in user_data.movieId]
        rankings_list.append(ranking)
    

    return rankings_list
