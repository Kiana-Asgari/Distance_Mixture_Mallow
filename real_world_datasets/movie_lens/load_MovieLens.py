import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from typing import Literal
import pandas as pd
import numpy as np


def load_and_return_ratings_movies(n_movies: int = 10, version: Literal["ml-25m", "ml-latest"] = "ml-latest", min_ratings: int = 1000):

    print(f"▶ Loading MovieLens {version}")
    
    # Download and extract dataset
    data_dir = _download_movielens(version)
    ratings, movies = _load_ratings_movies(data_dir)
    print(f"  Loaded {len(ratings):,} ratings from {ratings.userId.nunique():,} users")
    
    # Select movies that maximize user coverage
    print(f"▶ Selecting {n_movies} movies")
    selected_movie_ids = _select_movies(ratings, n_movies, min_ratings)
    
    # Get movie titles for display
    movie_titles = movies.set_index("movieId").loc[selected_movie_ids, "title"].tolist()
    print(f"\nSelected movies:")
    for i, (movie_id, title) in enumerate(zip(selected_movie_ids, movie_titles), 1):
        print(f"  {i}. {title}")
    
    # Build rankings for users who rated all selected movies
    print(f"\n▶ Building user rankings")
    rankings_list = _build_user_rankings(ratings, selected_movie_ids)
    print(f"  Found {len(rankings_list):,} users with complete rankings\n")
    
    return rankings_list


def _download_movielens(version: str) -> Path: #
    """Download and extract MovieLens dataset, using cache if available."""
    cache_dir = Path(__file__).parent / "cache_files" / "movielens"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_dir = cache_dir / version
    zip_path = cache_dir / f"{version}.zip"
    
    # Return cached data if available
    if extracted_dir.exists():
        print(f"  ✔ Using cached data")
        return extracted_dir
    
    # Download if needed
    if not zip_path.exists():
        url = f"https://files.grouplens.org/datasets/movielens/{version}.zip"
        print(f"  Downloading {version}...")
        urlretrieve(url, zip_path)
        print(f"  ✔ Download complete")
    
    # Extract
    print(f"  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)
    print(f"  ✔ Data ready")
    
    return extracted_dir


def _load_ratings_movies(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings and movies data, handling both CSV and DAT formats."""
    # Check which format is available
    if (data_dir / "ratings.csv").exists():
        # CSV format (ml-25m, ml-latest-small)
        ratings = pd.read_csv(
            data_dir / "ratings.csv",
            usecols=["userId", "movieId", "rating", "timestamp"]
        )
        movies = pd.read_csv(
            data_dir / "movies.csv",
            usecols=["movieId", "title"]
        )
    else:
        # DAT format with :: delimiter (ml-1m)
        ratings = pd.read_csv(
            data_dir / "ratings.dat",
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"]
        )
        movies = pd.read_csv(
            data_dir / "movies.dat",
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            usecols=[0, 1]
        )
    
    return ratings, movies


def _select_movies(ratings: pd.DataFrame, n_movies: int, min_ratings: int) -> list[int]:
    """
    Select movies using greedy algorithm to maximize number of users with complete ratings.
    Fully vectorized using numpy operations for handling millions of ratings.
    """
    # Get movies with enough ratings
    movie_counts = ratings.groupby("movieId").size()
    candidate_movies = movie_counts[movie_counts >= min_ratings].sort_values(ascending=False).index.tolist()
    
    if len(candidate_movies) < n_movies:
        raise ValueError(f"Only {len(candidate_movies)} movies have ≥{min_ratings} ratings (need {n_movies})")
    
    # FULLY VECTORIZED: Build user-movie boolean matrix
    top_candidates = candidate_movies[:100]
    ratings_subset = ratings[ratings.movieId.isin(top_candidates)].copy()
    
    # Create categorical codes for efficient indexing
    ratings_subset["movie_idx"] = pd.Categorical(
        ratings_subset["movieId"], 
        categories=top_candidates
    ).codes
    
    # Map user IDs to indices
    unique_users = ratings_subset["userId"].unique()
    user_to_idx = pd.Series(np.arange(len(unique_users)), index=unique_users)
    ratings_subset["user_idx"] = ratings_subset["userId"].map(user_to_idx)
    
    # Create boolean matrix: users × movies
    n_users = len(unique_users)
    n_candidates = len(top_candidates)
    user_movie_matrix = np.zeros((n_users, n_candidates), dtype=bool)
    user_movie_matrix[ratings_subset["user_idx"].values, ratings_subset["movie_idx"].values] = True
    
    # Greedy selection using vectorized operations
    selected_indices = []
    available_mask = np.ones(n_candidates, dtype=bool)
    user_mask = np.ones(n_users, dtype=bool)  # Users who rated all selected movies
    
    for step in range(n_movies):
        # Vectorized: compute user overlap for ALL remaining candidates at once
        # For each movie, count users who: (1) rated this movie AND (2) rated all selected movies
        overlap_counts = (user_movie_matrix[:, available_mask] & user_mask[:, np.newaxis]).sum(axis=0)
        
        # Find best movie
        best_idx_in_available = overlap_counts.argmax()
        best_count = overlap_counts[best_idx_in_available]
        
        if best_count == 0:
            raise ValueError(f"Cannot find {n_movies} movies with overlapping users")
        
        # Map back to original index
        available_indices = np.where(available_mask)[0]
        best_movie_idx = available_indices[best_idx_in_available]
        
        # Update state
        selected_indices.append(best_movie_idx)
        available_mask[best_movie_idx] = False
        user_mask &= user_movie_matrix[:, best_movie_idx]  # Keep only users who rated this movie
        
        best_movie_id = top_candidates[best_movie_idx]
        print(f"  Step {step + 1}: selected movie {best_movie_id} ({best_count} users with all movies so far)")
    
    # Convert indices back to movie IDs
    selected_movie_ids = [top_candidates[idx] for idx in selected_indices]
    return selected_movie_ids


def _build_user_rankings(ratings: pd.DataFrame, selected_movie_ids: list[int]) -> list[list[int]]:
    """
    Build rankings for users who rated all selected movies.
    
    Returns rankings as lists of 1-based indices (1 = best movie for that user).
    """
    # Filter to selected movies
    subset = ratings[ratings.movieId.isin(selected_movie_ids)].copy()
    
    # Keep only users who rated all movies
    users_with_all = subset.groupby("userId").movieId.nunique()
    complete_users = users_with_all[users_with_all == len(selected_movie_ids)].index
    subset = subset[subset.userId.isin(complete_users)]
    
    # VECTORIZED ranking construction
    # Create mapping from movieId to 1-based index
    movie_to_index = pd.Series(
        range(1, len(selected_movie_ids) + 1), 
        index=selected_movie_ids
    )
    
    # Map movie IDs to indices in one vectorized operation
    subset["movie_index"] = subset["movieId"].map(movie_to_index)
    
    # Sort by user, then by rating (desc), then by timestamp (asc)
    # This groups users and orders movies by preference in one pass
    subset = subset.sort_values(
        ["userId", "rating", "timestamp"], 
        ascending=[True, False, True]
    )
    
    # Extract rankings as a list of lists using groupby
    # This avoids the explicit loop over users
    rankings_list = (
        subset.groupby("userId", sort=False)["movie_index"]
        .apply(list)
        .tolist()
    )
    
    return rankings_list