import requests
from collections import defaultdict

# 1. This URL points to the GitHub API, which lists the files in the directory.
file_list_url = (
    "https://api.github.com/repos/"
    "n-boehmer/Collecting-Classifying-Analyzing-and-Using-Real-World-Elections"
    "/contents/complete/football%20week"
)

# 2. This is the base URL for the **raw** GitHub content of each .soc file.
# Make sure it has a trailing slash, and note that the space in "football week"
# is properly URL-encoded as "%20".
base_raw_url = (
    "https://raw.githubusercontent.com/"
    "n-boehmer/Collecting-Classifying-Analyzing-and-Using-Real-World-Elections/"
    "main/complete/football%20week/"
)

def fetch_file_list(limit=10):
    """
    Fetch the list of .soc files (up to `limit`) from the GitHub API URL.
    """
    response = requests.get(file_list_url)
    if response.status_code == 200:
        files = response.json()
        # Keep only .soc files
        soc_files = [file['name'] for file in files if file['name'].endswith('.soc')]
        return soc_files[:limit]
    else:
        print(f"Failed to fetch file list from {file_list_url}. "
              f"Status code: {response.status_code}")
        return []

def fetch_and_process_data(file_name):
    """
    Fetch and process a single .soc file by using the raw file URL.
    """
    # Build the raw file URL
    raw_file_url = base_raw_url + file_name

    response = requests.get(raw_file_url)
    if response.status_code == 200:
        data = response.text.splitlines()
        n = int(data[0].strip())

        # Lines [1 : n+1] contain team info in the format: "i, TeamName"
        teams = [line.split(',')[1].strip() for line in data[1:n+1]]

        # The next line after teams contains three integers (a, b, c)
        a, b, c = map(int, data[n+1].split(','))

        # Then, the next `a` lines contain the rankings (votes). Each line
        # has comma-separated integers followed by a blank last column, so
        # we slice off the last empty piece with [:-1].
        votes = [list(map(int, line.split(',')))[:-1] for line in data[n+2 : n+2+a]]

        return teams, votes
    else:
        print(f"Failed to fetch data from {raw_file_url} "
              f"(status code: {response.status_code}).")
        return None, None

def unify_teams_and_votes(file_names):
    """
    For each .soc file:
      - Create/extend a global mapping of team_name -> unified integer
      - Convert each file's votes to the unified numbering
    """
    team_mapping = {}
    unified_teams = []
    unified_votes = defaultdict(list)
    team_counter = 1

    for file_name in file_names:
        teams, votes = fetch_and_process_data(file_name)
        if not teams or not votes:
            continue

        # Map any new teams to a new integer
        for team in teams:
            if team not in team_mapping:
                team_mapping[team] = team_counter
                unified_teams.append(team)
                team_counter += 1

        # Convert the votes (indices) to the unified numbering
        for vote in votes:
            # Each 'vote' is a list of positions (e.g. [3,1,2])
            # They are 1-based indices for 'teams', so teams[vote[i] - 1] 
            # is the name. Then we look up that name in the team_mapping.
            mapped_vote = [team_mapping[teams[i - 1]] for i in vote]
            unified_votes[file_name].append(mapped_vote)

    return team_mapping, unified_teams, unified_votes

def load_data(limit=10):
    # 1. Get the file list from GitHub.
    file_names = fetch_file_list(limit)
    if not file_names:
        print("No .soc files found; exiting.")
        return

    # 2. Unify the teams and votes.
    team_mapping, unified_teams, all_votes = unify_teams_and_votes(file_names)

    # 3. Print out the mapping and the new votes.
    #print("Unified Team Mapping:")
    ##for team, number in team_mapping.items():
    #    print(f"{number}: {team}")

    #print("\nUnified Votes (by file):")
    #for file_name, votes in all_votes.items():
    #    print(f"\nFile: {file_name}")
    #    for vote in votes:
    #        print("  ", vote)
    return  unified_teams, all_votes
