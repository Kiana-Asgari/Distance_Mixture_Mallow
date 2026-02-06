"""Main data loader dispatcher for all datasets."""
from real_world_datasets.loaders.college_sport_loader import load_college_sports
from real_world_datasets.loaders.movie_lens_loader import load_movie_lens
from real_world_datasets.loaders.news_loader import load_news
from real_world_datasets.loaders.sushi_loader import load_sushi


def load_data(dataset_name: str, n_items: int = 10):
    """
    Load data from the specified dataset.
    
    Args:
        dataset_name: Name of dataset ('basketball', 'football', 'movie_lens', 'news', 'sushi')
        n_items: Number of items/teams to load
        
    Returns:
        List of rankings as lists of 1-based integer IDs
    """
    if dataset_name == 'sushi':
        return load_sushi()
    elif dataset_name == 'movie_lens':
        return load_movie_lens(n_movies=n_items)
    elif dataset_name == 'news':
        return load_news(n_items=n_items)
    elif dataset_name in ['basketball', 'football']:
        return load_college_sports(dataset_name=dataset_name, n_teams=n_items)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")