from __future__ import annotations

import io
import os
import zipfile
import numpy as np
import urllib.request
from tqdm import tqdm  # Add tqdm for progress bars


def load_sushi():
    download_sushi_data()
    data = 1 + read_sushi_10_order()
    return data


def read_sushi_10_order():
    # Path to the data file
    file_path = os.path.join("real_world_datasets","sushi_dataset", "sushi3-2016", "sushi3a.5000.10.order")
    
    # Initialize array to store the rankings
    rankings = np.zeros((5000, 10), dtype=int)
    
    # Read the file and extract rankings
    with open(file_path, 'r') as f:
        line_idx = 0
        for line in f:
            line = line.strip()
                
            # Parse the line
            parts = line.split(' ')
            values = [int(part) for part in parts]
            if len(values) < 10:
                continue
                
            # The actual rankings are from the 3rd to 12th values (indices 2-11)
            rankings[line_idx] = [int(values[i]) for i in range(2, 12)]
            line_idx += 1
            
            # Stop if we've read 5000 rankings
            if line_idx >= 5000:
                break
    
    return rankings


def download_sushi_data():
    url = "https://www.kamishima.net/asset/sushi3-2016.zip"
    saving_dir = "real_world_datasets/sushi_dataset"
    
    # Create directory if it doesn't exist
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
        
    # Check if data files already exist
    data_files = os.path.join(saving_dir, "sushi3-2016", "sushi3a.5000.10.order")
    if os.path.exists(data_files):
        print(f"Sushi data already exists in {saving_dir}")
        return
    
    # Download and extract the data
    print(f"Downloading sushi data from {url}...")
    
    # Open the URL and get file size
    with urllib.request.urlopen(url) as response:
        file_size = int(response.info().get('Content-Length', 0))
        
        # Create a progress bar for downloading
        progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading")
        
        # Download with progress updates
        zip_content = bytearray()
        chunk_size = 8192
        while True:
            buffer = response.read(chunk_size)
            if not buffer:
                break
            zip_content.extend(buffer)
            progress_bar.update(len(buffer))
        progress_bar.close()
    
    # Extract with progress bar
    print("Extracting files...")
    with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
        # Get list of files to extract
        file_list = zip_file.namelist()
        
        # Extract files with progress bar
        for file in tqdm(file_list, desc="Extracting"):
            zip_file.extract(file, saving_dir)
    
    print(f"Sushi data downloaded and extracted to {saving_dir}")
    