import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


def load_movie_lens(n_movies: int = 10, version: str = "ml-25m", min_ratings: int = 1000):
    """
    Load MovieLens dataset and return rankings for users who rated all selected movies.
    
    Args:
        n_movies: Number of movies to select
        version: MovieLens version (e.g., "ml-25m", "ml-1m")
        min_ratings: Minimum number of ratings a movie must have to be considered
    
    Returns:
        List of rankings, where each ranking is a list of movie indices (1-based)
    """
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


def _download_movielens(version: str = "ml-25m") -> Path:
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
    
    Greedy strategy: iteratively pick the movie that preserves the most users who have
    rated all movies selected so far.
    """
    # Get movies with enough ratings
    movie_counts = ratings.groupby("movieId").size()
    candidate_movies = movie_counts[movie_counts >= min_ratings].sort_values(ascending=False).index.tolist()
    
    if len(candidate_movies) < n_movies:
        raise ValueError(f"Only {len(candidate_movies)} movies have ≥{min_ratings} ratings (need {n_movies})")
    
    # Build index: movie_id -> set of users who rated it
    movie_to_users = {
        movie_id: set(ratings[ratings.movieId == movie_id].userId)
        for movie_id in candidate_movies[:100]  # Consider top 100 most-rated movies
    }
    
    # Greedy selection
    selected = []
    common_users = None
    
    for step in range(n_movies):
        best_movie = None
        best_count = -1
        best_users = None
        
        for movie_id in movie_to_users:
            if movie_id in selected:
                continue
            
            # Find users who rated this movie AND all previously selected movies
            users_for_this = movie_to_users[movie_id]
            if common_users is None:
                candidate_users = users_for_this
            else:
                candidate_users = common_users & users_for_this
            
            count = len(candidate_users)
            if count > best_count:
                best_count = count
                best_movie = movie_id
                best_users = candidate_users
        
        if best_movie is None or best_count == 0:
            raise ValueError(f"Cannot find {n_movies} movies with overlapping users")
        
        selected.append(best_movie)
        common_users = best_users
        print(f"  Step {step + 1}: selected movie {best_movie} ({best_count} users with all movies so far)")
    
    return selected


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
    
    # Build rankings for each user
    rankings_list = []
    movie_to_index = {movie_id: i + 1 for i, movie_id in enumerate(selected_movie_ids)}
    
    for user_id in complete_users:
        user_data = subset[subset.userId == user_id].copy()
        # Sort by rating (descending) then timestamp (ascending) for tiebreaking
        user_data = user_data.sort_values(["rating", "timestamp"], ascending=[False, True])
        # Convert movie IDs to 1-based indices
        ranking = [movie_to_index[movie_id] for movie_id in user_data.movieId]
        rankings_list.append(ranking)
    
    return rankings_list
