# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os
import numpy as np

def load_chess(file_path="chess/data"):
    # Create directory if it doesn't exist
    os.makedirs(file_path, exist_ok=True)
    
    # Set the file path for saving the CSV
    csv_path = os.path.join(file_path, "chess_rankings_1851_2001_quarterly.csv")
    
    # Check if file already exists
    if os.path.exists(csv_path):
        print(f"Using existing file at {csv_path}")
        df = pd.read_csv(csv_path, encoding='latin1')
    else:
        print(f"Downloading chess rankings to {csv_path}")
        # Download the raw CSV file from GitHub
        url = "https://raw.githubusercontent.com/JGravier/chessplayers/main/csv/ranking_chessplayers_1851_2001_quarterly.csv"
        
        # Download and read the CSV with encoding specified
        df = pd.read_csv(url, encoding='latin1')
        
        # Save a local copy
        df.to_csv(csv_path, index=False, encoding='latin1')
    
    # Print the head of the DataFrame
    print("Chess rankings data head:")
    print(df.head())
    data = process_chess_data(df)
    
    return df


def process_chess_data(chess_data):
    print('Processing chess data...')
    
    # Verify we have the expected columns
    if 'ranking' not in chess_data.columns or 'Player' not in chess_data.columns:
        print("Warning: Required columns 'rank' or 'name' not found!")
        print("Available columns:", chess_data.columns.tolist())
        return []
    
    # Initialize variables
    rankings = []
    names = []

    
    current_ranking = []
    
    # Loop through each row
    rank = 1
    for iter_ranking, row in chess_data.iterrows():
        rank += iter_ranking
        current_rank = row['ranking']
        name = row['Player']

        if name not in names:
            names.append(name)
        current_ranking.append(name)

        if current_rank == 1 and iter_ranking != 0:
            rankings.append(current_ranking)
            current_ranking = []
    
    print(f"Extracted {len(rankings)} ranking cycles with {len(names)} players")
    training_data = name_to_id(rankings, names, 20)
    #print(training_data[0])
    #print(training_data[1])
    print(len(training_data))
    return training_data

def name_to_id(rankings, names, num_players_to_keep):
    how_many_times_played = [0] * len(names)
    most_played_players = []
    
    for name in names:
        for ranking in rankings:
            if name in ranking:
                how_many_times_played[names.index(name)] += 1

    # Get indices of top 10 most frequent players
    top_10_indices = np.argsort(how_many_times_played)[-10:]
    
    # Convert numpy array to list and add to most_played_players
    most_played_players = top_10_indices.tolist()
    
    # Print the top 10 indices
    print(top_10_indices)
    
    # Print the names of the top 10 players using list comprehension
    top_10_names = [names[i] for i in top_10_indices]
    print(top_10_names)


    chosen_rankings = []
    for ranking in rankings:
        flag = True
        ranking_to_id=[]
        for player in ranking:
            if player in top_10_names:
                ranking_to_id.append(names.index(player))
                flag = False
        if flag:
            chosen_rankings.append(ranking_to_id)

    print(len(chosen_rankings))
    return how_many_times_played, most_played_players, chosen_rankings
