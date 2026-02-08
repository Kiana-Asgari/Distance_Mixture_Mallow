import numpy as np
import urllib.request
from collections import Counter
from pathlib import Path
import time

def download_f1_season(season_num, cache_dir):
    """
    Download a single F1 season file if not already cached.
    
    Args:
        season_num: Season file number (e.g., 1 for 00052-00000001.soi)
        cache_dir: Directory to cache the file
        
    Returns:
        Path to the downloaded file, or None if download fails
    """
    filename = f"00052-{season_num:08d}.soi"
    filepath = cache_dir / filename
    
    if filepath.exists():
        return filepath
    
    # Try to download from GitHub
    url = f"https://raw.githubusercontent.com/PrefLib/PrefLib-Data/main/datasets/00052%20-%20f1seasons/{filename}"
    
    try:
        print(f"  Downloading season {season_num}...", end=" ")
        urllib.request.urlretrieve(url, filepath)
        print("✓")
        time.sleep(0.1)  # Be nice to the server
        return filepath
    except Exception as e:
        print(f"✗ (skipped)")
        return None

def download_all_f1_seasons(max_seasons=100):
    """
    Download all available F1 season files.
    
    Args:
        max_seasons: Maximum number of season files to attempt downloading
        
    Returns:
        List of paths to successfully downloaded files
    """
    cache_dir = Path(__file__).parent / "cache_files" / "f1seasons"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading F1 seasons to {cache_dir}")
    
    downloaded_files = []
    consecutive_failures = 0
    
    for season_num in range(1, max_seasons + 1):
        filepath = download_f1_season(season_num, cache_dir)
        if filepath:
            downloaded_files.append(filepath)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            # If we get 3 consecutive failures, assume no more files
            if consecutive_failures >= 3:
                print(f"  Stopped after {consecutive_failures} consecutive failures")
                break
    
    print(f"✓ Downloaded/cached {len(downloaded_files)} season files\n")
    return downloaded_files

def parse_soi_file(filename):
    """
    Parse the PrefLib SOI (Strict Orders - Incomplete List) format file.
    Supports the new format with # comments and ALTERNATIVE NAME metadata.
    Returns a list of rankings where each ranking is a list of team names.
    """
    rankings_by_name = []
    team_id_to_name = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse metadata and build team_id_to_name from # ALTERNATIVE NAME lines
    for line in lines:
        line = line.strip()
        if line.startswith('# ALTERNATIVE NAME'):
            # Format: # ALTERNATIVE NAME 1: pozzi
            parts = line.split(':', 2)
            if len(parts) >= 2:
                # Extract the number from "# ALTERNATIVE NAME 1"
                name_part = parts[0].strip()
                team_id = int(name_part.split()[-1])
                team_name = parts[1].strip()
                team_id_to_name[team_id] = team_name
    
    # Parse rankings
    for line in lines:
        line = line.strip()
        
        # Skip comments and empty lines
        if line.startswith('#') or not line:
            continue
        
        # Parse actual rankings in format: "1: 71,22,23,57,..."
        # where the first number is the count of voters with this ranking
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                # voter_count = int(parts[0].strip())  # Number before colon
                ranking_str = parts[1].strip()
                
                # Remove curly braces for ties (we'll just take them in order)
                ranking_str = ranking_str.replace('{', '').replace('}', '')
                team_ids = [int(x.strip()) for x in ranking_str.split(',') if x.strip()]
                
                # Convert team IDs to names
                ranking = [team_id_to_name[tid] for tid in team_ids if tid in team_id_to_name]
                
                if ranking:
                    rankings_by_name.append(ranking)
    
    return rankings_by_name

def choose_most_common_teams(rankings_by_name, n_teams_to_keep):
    """
    Choose n_teams_to_keep based on the most commonly participating teams in the rankings.
    """
    # Flatten the list of rankings into a single list of team names
    all_teams = [team for ranking in rankings_by_name for team in ranking]
    
    # Count occurrences of each team
    team_counts = Counter(all_teams)
    selected_teams = [team for team, count in team_counts.most_common(n_teams_to_keep)]
    print(f'F1 chosen teams (the first 5 out of {len(selected_teams)}):', selected_teams[:5])
    print(f'Total unique teams before filtering: {len(team_counts)}')
    return selected_teams

def filter_rankings(rankings_by_name, selected_teams):
    """
    Filter rankings to only include selected teams and convert to integer indices.
    Only keep rankings that contain all selected teams.
    """
    # Create a mapping from team name to index
    team_to_idx = {team: idx for idx, team in enumerate(selected_teams)}
    
    filtered_rankings = []
    
    for ranking in rankings_by_name:
        # Filter to only selected teams
        filtered_ranking = [team for team in ranking if team in team_to_idx]
        
        # Only keep if we have all selected teams in this ranking
        if len(filtered_ranking) == len(selected_teams):
            # Convert to indices
            indexed_ranking = [team_to_idx[team] for team in filtered_ranking]
            filtered_rankings.append(indexed_ranking)
    
    return filtered_rankings

def load_f1_rankings(n_teams_to_keep=100, max_seasons=100):
    """
    Main function to load F1 rankings from all available seasons and return as a 2D numpy array.
    
    Args:
        n_teams_to_keep: Number of most common teams to keep (default: 100)
        max_seasons: Maximum number of season files to download (default: 100)
    
    Returns:
        2D numpy array where each row is a complete ranking (integers 0 to n_teams_to_keep-1)
    """
    # Download all available season files
    season_files = download_all_f1_seasons(max_seasons)
    
    if not season_files:
        print("Error: No season files downloaded")
        return np.array([], dtype=np.int32)
    
    # Parse all season files and combine rankings
    print(f"Parsing {len(season_files)} season files...")
    all_rankings_by_name = []
    
    for filepath in season_files:
        rankings = parse_soi_file(filepath)
        all_rankings_by_name.extend(rankings)
    
    print(f"✓ Total rankings loaded from all seasons: {len(all_rankings_by_name)}")
    
    # Choose most common teams across all seasons
    print(f"\nSelecting top {n_teams_to_keep} most common teams across all seasons...")
    selected_teams = choose_most_common_teams(all_rankings_by_name, n_teams_to_keep)
    
    # Filter rankings to only include selected teams
    print("\nFiltering rankings to include only selected teams...")
    filtered_rankings = filter_rankings(all_rankings_by_name, selected_teams)
    print(f"✓ Rankings with all {n_teams_to_keep} teams: {len(filtered_rankings)}")
    
    # Convert to numpy array
    if filtered_rankings:
        rankings_array = np.array(filtered_rankings, dtype=np.int32)
        print(f"✓ Final array shape: {rankings_array.shape}")
        print(f"  (rows={rankings_array.shape[0]} rankings, cols={rankings_array.shape[1]} teams)\n")
        return rankings_array
    else:
        print("\nWarning: No complete rankings found with all selected teams.")
        print("This might happen if teams don't all appear together in any single ranking.")
        print("Consider reducing n_teams_to_keep or using a different filtering strategy.")
        return np.array([], dtype=np.int32)

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("F1 Rankings Loader - All Seasons Combined")
    print("="*70 + "\n")
    
    # Load the F1 rankings with top teams across all seasons
    rankings_array = load_f1_rankings(n_teams_to_keep=10)
    
    if rankings_array.size > 0:
        print("="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"Shape: {rankings_array.shape}")
        print(f"Number of rankings: {rankings_array.shape[0]:,}")
        print(f"Number of teams per ranking: {rankings_array.shape[1]}")
        print(f"\nFirst 5 rankings:")
        print(rankings_array[:5])
        print(f"\nLast 5 rankings:")
        print(rankings_array[-5:])
        print(f"\nData type: {rankings_array.dtype}")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("Try with fewer teams (e.g., n_teams_to_keep=5)")
        print("="*70)
        
        # Retry with fewer teams
        rankings_array = load_f1_rankings(n_teams_to_keep=5)
        if rankings_array.size > 0:
            print(f"\nWith 5 teams - Shape: {rankings_array.shape}")