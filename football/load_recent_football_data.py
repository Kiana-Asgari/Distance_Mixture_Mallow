import kagglehub
import os
import pandas as pd



def load_football_data(n_teams_to_keep=10):
    teams_to_keep = [
                    ' LSU                 ',
                    ' Clemson             ',
                    ' Ohio St             ',
                    ' Georgia             ',
                    ' Oregon              ',
                    ' Florida             ',
                    ' Oklahoma            ',
                    ' Alabama             ',
                    ' Penn St             ',
                    ' Minnesota           '
                    ][:n_teams_to_keep]
    all_team_names, rankings_by_name = fetch_recent_football_data()
    selected_rankings = choose_top_teams(rankings_by_name, teams_to_keep)
    selected_rankings_by_id = rankings_by_id(selected_rankings, teams_to_keep)
    print(f"Selected rankings: {len(selected_rankings)}, each of length {len(selected_rankings[0])}")
    print(f"the first ranking id is: {selected_rankings_by_id[0]}")
    print(f"the first ranking name is: {selected_rankings[0]}")
    return selected_rankings_by_id

def fetch_recent_football_data():
    # Download latest version
    path = kagglehub.dataset_download("masseyratings/rankings")
    
    # Read and print heads of specific files
    target_files = ["cf2019.csv", "cf2021.csv", "cf2020.csv"]
    rankings_by_name = []
    all_team_names = []

    for target_file in target_files:
        file_path = os.path.join(path, target_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            current_rankings, current_team_names = process_recent_football_data(df)
            rankings_by_name.extend(current_rankings)
            all_team_names.extend(current_team_names)
        else:
            print(f"\nFile not found: {target_file}")
    print(f"number of full rankings: {len(rankings_by_name)}")


    return all_team_names, rankings_by_name

def process_recent_football_data(df):
    rankings_by_name = []
    all_team_names = []
    current_ranking = []
    for index, row in df.iterrows():
        rank = row[-1]
        team_name = row[2]
        if team_name not in all_team_names:
            all_team_names.append(team_name)
        ## New week:
        if rank == 1 and len(current_ranking) > 10:
            rankings_by_name.append(current_ranking)
            current_ranking = []

            
        current_ranking.append(team_name)

    return rankings_by_name, all_team_names



def choose_top_teams(rankings_by_name, teams_to_keep):
    selected_rankings = []
    for rankings in rankings_by_name:
        current_ranking = []
        for team_name in rankings:
            if team_name in teams_to_keep:
                current_ranking.append(team_name)
        if len(current_ranking) == len(teams_to_keep):
            selected_rankings.append(current_ranking)
    return selected_rankings


def rankings_by_id(rankings_by_name, teams_to_keep):
    rankings_by_id = []

    for rankings in rankings_by_name:
        current_ranking = []
        for team_name in rankings:
            if team_name in teams_to_keep:
                current_ranking.append(teams_to_keep.index(team_name) + 1) #
        if len(current_ranking) == len(teams_to_keep):
            rankings_by_id.append(current_ranking)
    return rankings_by_id
