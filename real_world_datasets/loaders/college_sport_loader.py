import kagglehub
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from real_world_datasets.config import TOP_TEAMS, FILES


def load_college_sports(dataset_name, n_teams):
    """Load college sports dataset and return rankings as integer IDs."""
    print('loading data', dataset_name)
    n_teams_to_keep = n_teams
    
    # Fetch raw data from Kaggle
    all_teams, rankings = _fetch_data(dataset_name)
    
    # Select which teams to keep
    top_teams = TOP_TEAMS.get(dataset_name, [])
    
    if n_teams_to_keep <= len(top_teams):
        teams_to_keep = top_teams[:n_teams_to_keep]
    elif dataset_name == 'football':
        teams_to_keep = _most_common_teams(rankings, n_teams_to_keep)
    elif dataset_name == 'basketball':
        teams_to_keep = all_teams[:n_teams_to_keep]
    else:
        raise ValueError(f"Unknown dataset {dataset_name!r} (choose from {list(FILES)})")
    
    # Filter rankings to only include selected teams
    filtered_rankings = _filter_rankings(rankings, teams_to_keep)
    
    # Convert team names to integer IDs
    rankings_as_ids = _convert_to_ids(filtered_rankings, teams_to_keep)
    
    return rankings_as_ids


def _fetch_data(dataset_name):
    """Download and parse college sports data from Kaggle."""
    if dataset_name not in FILES:
        raise ValueError(f"Unknown dataset {dataset_name!r} (choose from {list(FILES)})")
    
    # Download dataset from KaggleHub
    data_dir = Path(kagglehub.dataset_download("masseyratings/rankings"))
    
    all_rankings = []
    all_teams = []
    
    for filename in tqdm(FILES[dataset_name], desc=f"Fetching {dataset_name} dataset. This may take a while..."):
        filepath = data_dir / filename
        if not filepath.is_file():
            tqdm.write(f"⚠️  missing: {filename}")
            continue
        
        # Process each CSV file
        df = pd.read_csv(filepath)
        rankings, teams = _parse_weekly_rankings(df)
        all_rankings.extend(rankings)
        all_teams.extend(teams)
    
    print(f"✓ {len(all_rankings):,} ranking rows • {len(all_teams):,} team names")
    return all_teams, all_rankings


def _parse_weekly_rankings(df, rank_col=-1, team_col=2, min_size=10):
    """Split CSV into weekly rankings and collect unique team names."""
    team_names = df.iloc[:, team_col].to_numpy()
    ranks = df.iloc[:, rank_col].to_numpy()
    
    # Find where each week starts (rank == 1)
    week_starts = np.flatnonzero(ranks == 1)
    week_ends = np.r_[week_starts[1:], len(team_names)]
    
    # Extract rankings for each week (keep only complete weeks)
    rankings = [
        team_names[start:end].tolist()
        for start, end in zip(week_starts, week_ends)
        if end - start > min_size
    ]
    
    # Get unique teams preserving order
    unique_teams = list(dict.fromkeys(team_names))
    
    return rankings, unique_teams


def _filter_rankings(rankings, teams_to_keep):
    """Keep only rankings that contain all the selected teams."""
    team_set = set(teams_to_keep)
    filtered = []
    
    for ranking in rankings:
        # Extract only the teams we want to keep
        filtered_ranking = [team for team in ranking if team in team_set]
        
        # Only keep rankings that have all selected teams
        if len(filtered_ranking) == len(teams_to_keep):
            filtered.append(filtered_ranking)
    
    return filtered


def _convert_to_ids(rankings, teams_to_keep):
    """Convert team names to 1-based integer IDs."""
    # Create mapping from team name to ID (1-indexed)
    team_to_id = {name: i + 1 for i, name in enumerate(teams_to_keep)}
    
    # Convert all rankings to use IDs instead of names
    rankings_as_ids = [
        [team_to_id[team] for team in ranking]
        for ranking in rankings
        if all(team in team_to_id for team in ranking)
    ]
    
    return rankings_as_ids


def _most_common_teams(rankings, n_teams):
    """Select the n most frequently appearing teams across all rankings."""
    # Flatten all rankings into a single list
    all_teams = [team for ranking in rankings for team in ranking]
    
    # Count occurrences and select top n
    team_counts = Counter(all_teams)
    selected = [team for team, _ in team_counts.most_common(n_teams)]
    
    print(f'Choosing {n_teams} most common teams. First 5: {selected[:5]}')
    return selected