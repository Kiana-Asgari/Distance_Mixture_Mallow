import pandas as pd
import numpy as np
from pathlib import Path

def load_news_data(n_items_to_keep=None):
    """
    Load news rankings from the full_rankings.csv file.
    
    Args:
        n_items_to_keep (int, optional): Number of items to keep from the rankings.
                                        If None, keeps all items (10).
    
    Returns:
        list: List of rankings, where each ranking is a list of item IDs (1-based indexing).
    """
    # Path to the CSV file
    csv_path = Path(__file__).parent / "full_rankings.csv"
    
    # Read the CSV file
    df = pd.read_csv(csv_path, header=None)
    
    # Process each row to extract rankings
    rankings = []
    for _, row in df.iterrows():
        # Get the first (and only) column value
        ranking_str = row.iloc[0]
        
        # Parse the comma-separated string and clean up whitespace
        ranking_items = [item.strip() for item in ranking_str.split(',')]
        
        # Convert item IDs from "id1", "id2", etc. to 1, 2, etc.
        ranking_ids = []
        for item in ranking_items:
            if item.startswith('id'):
                try:
                    item_id = int(item[2:])  # Remove "id" prefix and convert to int
                    ranking_ids.append(item_id)
                except ValueError:
                    # Skip invalid items
                    continue
        
        # Only keep rankings that have the expected number of items
        if len(ranking_ids) == 10:  # All rankings should have 10 items
            rankings.append(ranking_ids)
    
    # If n_items_to_keep is specified, truncate each ranking
    if n_items_to_keep is not None and n_items_to_keep < 10:
        rankings = [ranking[:n_items_to_keep] for ranking in rankings]
    
    print(f"✓ Loaded {len(rankings):,} news rankings")
    if rankings:
        print(f"✓ Each ranking contains {len(rankings[0])} items")
    print('first ranking', rankings[0])
    
    return rankings

def load_news_data_by_name(n_items_to_keep=None):
    """
    Load news rankings and return them with item names instead of IDs.
    This is similar to the college sports data format.
    
    Args:
        n_items_to_keep (int, optional): Number of items to keep from the rankings.
                                        If None, keeps all items (10).
    
    Returns:
        list: List of rankings, where each ranking is a list of item names.
    """
    # Path to the CSV file
    csv_path = Path(__file__).parent / "full_rankings.csv"
    
    # Read the CSV file
    df = pd.read_csv(csv_path, header=None)
    
    # Process each row to extract rankings
    rankings = []
    for _, row in df.iterrows():
        # Get the first (and only) column value
        ranking_str = row.iloc[0]
        
        # Parse the comma-separated string and clean up whitespace
        ranking_items = [item.strip() for item in ranking_str.split(',')]
        
        # Only keep rankings that have the expected number of items
        if len(ranking_items) == 10:  # All rankings should have 10 items
            rankings.append(ranking_items)
    
    # If n_items_to_keep is specified, truncate each ranking
    if n_items_to_keep is not None and n_items_to_keep < 10:
        rankings = [ranking[:n_items_to_keep] for ranking in rankings]
    
    print(f"✓ Loaded {len(rankings):,} news rankings")
    if rankings:
        print(f"✓ Each ranking contains {len(rankings[0])} items")
    
    return rankings
