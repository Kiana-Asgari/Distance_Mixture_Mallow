import numpy as np
from numpy.random import default_rng
from tabulate import tabulate
from real_world_datasets.config import MODEL_LABEL, MODEL_NICE_NAME, METRICS_NICE_NAMES

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
    train_size = 700#int(training_ratio * len(train_pool_indices))
    test_size = 150#int(training_ratio * len(test_pool_indices))
    
    train_indices = rng.choice(train_pool_indices, size=train_size, replace=False)
    test_indices = rng.choice(test_pool_indices, size=test_size, replace=False)
    
    # Split data into training and testing sets
    train_data = sport_data_array[train_indices]
    test_data = sport_data_array[test_indices]
    
    return train_data, test_data, train_indices, test_indices



####################################
# Table builder
####################################
def make_table(trials):
    rows = []

    # α and β belong to Original L1 only
    for p in ('alpha', 'beta'):
        arr = _fetch(trials, p)
        rows.append([f"Estimated {p}"] + [_col(arr)] + ['--', '--'])

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
                tablefmt='latex'))



def _col(values, scale=1):
    """mean ± std, optionally scaled (e.g. to %)"""
    return f"{np.mean(values)*scale:.3f} (± {np.std(values)*scale:.3f})"


def _fetch(trials, key):
    return np.array([t[key] for t in trials])
