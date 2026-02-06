"""
Data loaders for all real-world ranking datasets.
"""
from real_world_datasets.loaders.college_sport_loader import load_college_sports
from real_world_datasets.loaders.movie_lens_loader import load_movie_lens
from real_world_datasets.loaders.news_loader import load_news
from real_world_datasets.loaders.sushi_loader import load_sushi
from real_world_datasets.loaders.data_loader import load_data

__all__ = [
    'load_college_sports',
    'load_movie_lens',
    'load_news',
    'load_sushi',
    'load_data',
]
