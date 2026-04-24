"""Dataset loaders that mirror the main pipeline conventions."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from real_world_datasets.utils import (
    chronologically_train_split,
    train_split,
)


# Datasets where we use a chronological split (sports), vs random 80/20 (others).
SPORTS = {"football", "basketball", "baseball"}
OTHERS = {"sushi", "movie_lens", "news"}


class DatasetUnavailable(RuntimeError):
    """Raised when a dataset cannot be loaded (e.g., needs auth or network)."""


def load_dataset(name: str, n_items: int):
    """Returns the data array (1-based rankings, shape m x n).

    Raises DatasetUnavailable when the dataset cannot be fetched from the
    current environment (e.g., Kaggle credentials missing).
    """
    try:
        if name == "sushi":
            from real_world_datasets.sushi_dataset.load_data import load_sushi
            return load_sushi()
        if name == "movie_lens":
            from real_world_datasets.movie_lens.load_MovieLens import (
                load_and_return_ratings_movies,
            )
            return load_and_return_ratings_movies(n_movies=n_items)
        if name == "news":
            from real_world_datasets.news.load_news import load_news_data
            return np.asarray(load_news_data(n_items_to_keep=n_items))
        if name in SPORTS:
            from real_world_datasets.college_sports.load_data import load_data
            return load_data(dataset_name=name, n_teams_to_keep=n_items)
    except Exception as exc:
        raise DatasetUnavailable(f"Cannot load {name} (n={n_items}): {exc}") from exc
    raise ValueError(f"Unknown dataset {name}")


def split(name: str, data, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if name in SPORTS:
        train, test, *_ = chronologically_train_split(np.asarray(data), seed)
    else:
        train, test, *_ = train_split(np.asarray(data), 0.7, seed)
    return np.asarray(train), np.asarray(test)


def all_dataset_specs():
    """List of (dataset_name, n_items) used throughout the reviewer response."""
    return [
        ("football", 10),
        ("football", 100),
        ("basketball", 10),
        ("basketball", 100),
        ("baseball", 10),
        ("baseball", 100),
        ("sushi", 10),
        ("movie_lens", 10),
        ("movie_lens", 50),
        ("movie_lens", 100),
        ("news", 10),
    ]


def small_dataset_specs():
    """n=10 datasets only -- used for the experiments that require n<=10."""
    return [(n, k) for (n, k) in all_dataset_specs() if k == 10]
