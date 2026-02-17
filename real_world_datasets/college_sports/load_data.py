import kagglehub
import os
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from pathlib import Path
from real_world_datasets.config import TOP_TEAMS, FILES
from collections import Counter

def load_data(n_teams_to_keep, dataset_name):
    print('loading data', dataset_name)
    top_10 = top_10_teams(dataset_name)
    all_team_names, rankings_by_name = fetch_data(dataset_name)
    breakpoint()


    if n_teams_to_keep <= len(top_10):
        teams_to_keep = top_10[:n_teams_to_keep]
    elif dataset_name == 'football':
        print(f'choosing {n_teams_to_keep} most common teams for {dataset_name}')
        teams_to_keep = chose_most_common_teams(rankings_by_name, n_teams_to_keep)
    elif dataset_name == 'basketball' or dataset_name == 'baseball':
        teams_to_keep = all_team_names[:n_teams_to_keep]
    else:
        raise ValueError(f"Unknown dataset {dataset_name!r} (choose from {list(FILES)})")
   
    selected_rankings = choose_top_teams(rankings_by_name, teams_to_keep)
    selected_rankings_by_id = rankings_by_id(selected_rankings, teams_to_keep)

    return selected_rankings_by_id


def fetch_data(ds):
    if ds not in FILES:
        raise ValueError(f"Unknown dataset {ds!r} (choose from {list(FILES)})")
    
    # KaggleHub shows its own download progress:
    root = Path(kagglehub.dataset_download("masseyratings/rankings"))

    ranks, teams = [], []
    for fname in tqdm(FILES[ds], desc=f"Fetching {ds} dataset. This may take a while..."):
        fpath = root / fname
        if not fpath.is_file():
            tqdm.write(f"⚠️  missing: {fname}")
            continue

        r, t = process_data(pd.read_csv(fpath))  # <- your helper
        ranks += r
        teams += t

    print(f"✓ {len(ranks):,} ranking rows • {len(teams):,} team names")
    return teams, ranks


# ------------------------------------------------------------------
# 1) split the CSV into "weekly" rankings + collect unique teams
# ------------------------------------------------------------------
def process_data(df, *, rank_col=-1, team_col=2, min_size=10):
    names = df.iloc[:, team_col].to_numpy()
    ranks = df.iloc[:, rank_col].to_numpy()

    # rows where a new week starts (rank == 1)
    starts = np.flatnonzero(ranks == 1)
    ends   = np.r_[starts[1:], names.size]               # next start, or EOF

    rankings = [names[s:e].tolist()                      # slice once per week
                for s, e in zip(starts, ends)
                if e - s > min_size]                     # keep only full weeks

    all_teams = list(dict.fromkeys(names))               # order-preserving uniques
    return rankings, all_teams


# ------------------------------------------------------------------
# 2) keep only rankings that contain the complete wanted set
# ------------------------------------------------------------------
def choose_top_teams(rankings, keep):
    keep_set = set(keep)
    out = []
    for r in rankings:
        picked = [n for n in r if n in keep_set]         # O(len(r))
        if len(picked) == len(keep):                     # every team present
            out.append(picked)
    return out


# ------------------------------------------------------------------
# 3) convert those rankings to 1-based IDs once, via a dict lookup
# ------------------------------------------------------------------
def rankings_by_id(rankings, keep):
    id_map = {name: i + 1 for i, name in enumerate(keep)}  # name → id
    return [
        [id_map[n] for n in r]                             # O(len(r))
        for r in rankings
        if all(n in id_map for n in r)                     # safety guard
    ]


##########################################
# Top 10 teams for each dataset
##########################################
def top_10_teams(dataset_name):
    """Return the top teams for the specified dataset."""
    if dataset_name in TOP_TEAMS:
        return TOP_TEAMS[dataset_name]
    return []

def chose_most_common_teams(rankings_by_name, n_teams_to_keep):
    """
    Choose n_teams_to_keep based on the most commonly participating teams in the rankings.
    """
    # Flatten the list of rankings into a single list of team names
    all_teams = [team for ranking in rankings_by_name for team in ranking]
    
    # Count occurrences of each team
    team_counts = Counter(all_teams)
    selected_teams = [team for team, count in team_counts.most_common(n_teams_to_keep)]
    print(f'football chosen teams (the first 5 out of {len(all_teams)}):', selected_teams[:5])
    return selected_teams