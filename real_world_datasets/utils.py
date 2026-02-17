import numpy as np
from numpy.random import default_rng
from tabulate import tabulate
from real_world_datasets.config import MODEL_LABEL, MODEL_NICE_NAME, METRICS_NICE_NAMES


def check_one_besed_index(pis: np.ndarray) -> bool:
    pis = np.asarray(pis)
    based_index = pis[0].min()
    pis = pis - based_index + 1
    for perm in pis:
        if perm.min() != 1:
            raise ValueError("The permutations are not one-based indexed")
    return pis

def check_zero_based_index(pis: np.ndarray) -> bool:
    pis = check_one_besed_index(pis) - 1
    for perm in pis:
        if perm.min() != 0:
            raise ValueError("The permutations are not zero-based indexed")
    return pis




def train_split(sushi_data, train_ratio, seed):
    # Randomly select indices for training using the RNG
    # Initialize random number generator with seed
    train_size = int(len(sushi_data) * train_ratio)
    rng = default_rng(seed)
    all_indices = np.arange(len(sushi_data))
    rng.shuffle(all_indices)
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:]
    
    # Split data into training and testing sets
    # Handle both numpy arrays and lists
    if isinstance(sushi_data, np.ndarray):
        train_data = sushi_data[train_indices]
        test_data = sushi_data[test_indices]
    else:
        # For lists or other iterables, use list comprehension
        train_data = [sushi_data[i] for i in train_indices]
        test_data = [sushi_data[i] for i in test_indices]
    
    return np.asarray(train_data), np.asarray(test_data), np.asarray(train_indices), np.asarray(test_indices)




def chronologically_train_split(sport_data, seed=None):
    # Initialize random number generator with seed
    rng = default_rng(seed)
    
    # Convert sport_data to numpy array if it's not already
    sport_data_array = np.asarray(sport_data)
    n_samples = len(sport_data_array)
    
    # Use 80% of data for potential training, 20% for potential testing
    train_test_split = int(n_samples * 0.7)
    
    # All indices before the split point are potential training data
    train_pool_indices = np.arange(train_test_split)
    
    # All indices after the split point are potential test data
    test_pool_indices = np.arange(train_test_split, n_samples)
    
    # Randomly sample train_size indices from the training pool
    # Use the smaller of desired size or available pool size
    desired_train_size = 700
    desired_test_size = 150
    
    train_size = min(desired_train_size, len(train_pool_indices)-100)
    test_size = min(desired_test_size, len(test_pool_indices)-25)
    
    train_indices = rng.choice(train_pool_indices, size=train_size, replace=False)
    test_indices = rng.choice(test_pool_indices, size=test_size, replace=False)
    
    # Split data into training and testing sets
    train_data = sport_data_array[train_indices]
    test_data = sport_data_array[test_indices]
    
    return train_data, test_data, train_indices, test_indices


def convert_numpy_to_native(obj):
    """Convert numpy arrays to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    return obj




####################################
# Table builder
####################################
def make_table(trials):
    rows = []

    # α and β belong to Original L1 (ML), Mallows Footrule (footrule), and Mallows Spearman (spearman)
    for p in ('alpha', 'beta'):
        arr = _fetch(trials, p)
        arr_footrule = _fetch(trials, f"{p}_footrule")
        arr_spearman = _fetch(trials, f"{p}_spearman")
        
        # Debug: Check if spearman values are being found
        if len(trials) > 0 and f"{p}_spearman" in trials[0]:
            print(f"Found {p}_spearman values: {arr_spearman}")
        
        # Debug: Print the first few values to check if they're being found
        if len(trials) > 0:
            print(f"Debug: {p}_spearman values found: {arr_spearman[:3]}")  # Show first 3 values
        
        # Create row with alpha/beta for ML, footrule, and spearman models, '--' for others
        row = [f"Estimated {p}"]
        for m in MODEL_LABEL:
            if MODEL_LABEL[m] == 'ML':  # L_α-Mallows
                row.append(_col(arr))
            elif MODEL_LABEL[m] == 'footrule':  # L₁-Mallows
                row.append(_col(arr_footrule))
            elif MODEL_LABEL[m] == 'spearman':  # L₂-Mallows
                row.append(_col(arr_spearman))
            else:
                row.append('--')
        rows.append(row)

    for name, (template, label, scale) in METRICS_NICE_NAMES.items():
        if name == 'top_k':           # need columns for k = 1 and 5
            for k in (1, 5):
                rows.append([label.format(k=k)] + [
                    _col(_fetch(trials,
                            template.format(MODEL_LABEL[m]))[:, k - 1], scale)
                    for m in MODEL_LABEL])
        else:
            rows.append([label] + [
                _col(_fetch(trials, template.format(MODEL_LABEL[m])), scale)
                for m in MODEL_LABEL])

    print(tabulate(rows, headers=['Metric'] + list(MODEL_NICE_NAME.values()),
                tablefmt='fancy_grid'))

def _col(values, scale=1):
    """mean ± std, optionally scaled (e.g. to %)"""
    return f"{100*np.mean(values)*scale:.1f} (± {100*np.std(values)*scale:.1f})"


def _fetch(trials, key):
    try:
        return np.array([t[key] for t in trials])
    except KeyError:
        # If key doesn't exist, return array of zeros with same length as trials
        return np.zeros(len(trials))
