"""Sushi preference dataset loader."""
import io
import zipfile
from pathlib import Path
import urllib.request

import numpy as np
from tqdm import tqdm


def load_sushi():
    """Load sushi preference dataset."""
    _download_sushi()
    data = 1 + _read_sushi_rankings()
    return data.tolist()


def _download_sushi():
    """Download and extract sushi dataset if not already present."""
    url = "https://www.kamishima.net/asset/sushi3-2016.zip"
    cache_dir = Path(__file__).parent / "cache_files" / "sushi"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = cache_dir / "sushi3-2016" / "sushi3a.5000.10.order"
    if data_file.exists():
        return
    
    print(f"Downloading sushi dataset...")
    
    # Download with progress bar
    with urllib.request.urlopen(url) as response:
        file_size = int(response.info().get('Content-Length', 0))
        progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading")
        
        zip_content = bytearray()
        while True:
            buffer = response.read(8192)
            if not buffer:
                break
            zip_content.extend(buffer)
            progress_bar.update(len(buffer))
        progress_bar.close()
    
    # Extract
    print("Extracting...")
    with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
        for file in tqdm(zip_file.namelist(), desc="Extracting"):
            zip_file.extract(file, cache_dir)
    
    print(f"✓ Sushi data ready")


def _read_sushi_rankings() -> np.ndarray:
    """Read sushi rankings from the data file."""
    file_path = Path(__file__).parent / "cache_files" / "sushi" / "sushi3-2016" / "sushi3a.5000.10.order"
    rankings = np.zeros((5000, 10), dtype=int)
    
    with open(file_path, 'r') as f:
        for line_idx, line in enumerate(f):
            if line_idx >= 5000:
                break
            
            parts = line.strip().split(' ')
            values = [int(part) for part in parts]
            
            if len(values) >= 12:
                # Rankings are from indices 2-11
                rankings[line_idx] = [values[i] for i in range(2, 12)]
    
    return rankings
