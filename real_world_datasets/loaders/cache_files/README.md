# Cache Files Directory

This directory contains all cached dataset files for the real-world data loaders.

## Structure

```
cache_files/
├── movielens/          # MovieLens dataset cache
│   ├── ml-25m/         # Full 25M ratings dataset (when downloaded)
│   ├── ml-1m/          # 1M ratings dataset (when downloaded)
│   ├── ml-latest-small/# Small dataset for testing
│   └── *.zip           # Downloaded zip files
├── news/               # News rankings dataset
│   └── full_rankings.csv
└── sushi/              # Sushi preference dataset
    └── sushi3-2016/    # Downloaded sushi dataset
```

## Purpose

All dataset loaders use this unified cache directory to:
- **Avoid redundant downloads**: Once downloaded, data is reused
- **Centralized storage**: All cached data in one location
- **Easy cleanup**: Delete this directory to clear all caches

## Cache Behavior

### MovieLens
- Downloads from `https://files.grouplens.org/datasets/movielens/`
- Caches both zip files and extracted directories
- Supports multiple versions (ml-25m, ml-1m, ml-latest-small)

### News
- Static CSV file included in repository
- No download required

### Sushi
- Downloads from `https://www.kamishima.net/asset/sushi3-2016.zip`
- Extracts to `sushi/sushi3-2016/`
- Contains 5000 user rankings over 10 sushi types

### College Sports
- Downloads from Kaggle via `kagglehub`
- Kaggle manages its own cache (not in this directory)

## Git Ignore

This directory is typically ignored in `.gitignore` except for static files:
- `news/full_rankings.csv` is tracked (static dataset)
- All other contents are gitignored (downloaded/cached data)

## Cleanup

To clear all cached datasets:
```bash
rm -rf real_world_datasets/loaders/cache_files/movielens
rm -rf real_world_datasets/loaders/cache_files/sushi
# Keep news/full_rankings.csv as it's a static file
```

Or clear specific datasets:
```bash
# Clear MovieLens cache
rm -rf real_world_datasets/loaders/cache_files/movielens

# Clear Sushi cache
rm -rf real_world_datasets/loaders/cache_files/sushi
```
