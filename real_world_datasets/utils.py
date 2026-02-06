import numpy as np
import pandas as pd
from pathlib import Path
from numpy.random import default_rng
from tabulate import tabulate
from real_world_datasets.config import MODEL_LABEL, MODEL_NICE_NAME, METRICS_NICE_NAMES



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
    train_size = 700#int(training_ratio * len(train_pool_indices))
    test_size = 150#int(training_ratio * len(test_pool_indices))
    
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
    """Build and print a formatted table of trial results."""
    rows = []

    # Add alpha and beta rows for Mallows models
    for p in ('alpha', 'beta'):
        arr = _fetch(trials, p)
        arr_footrule = _fetch(trials, f"{p}_footrule")
        arr_spearman = _fetch(trials, f"{p}_spearman")
        
        row = [f"Estimated {p}"]
        for m in MODEL_LABEL:
            if MODEL_LABEL[m] == 'ML':
                row.append(_col(arr))
            elif MODEL_LABEL[m] == 'footrule':
                row.append(_col(arr_footrule))
            elif MODEL_LABEL[m] == 'spearman':
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


####################################
# Results Printing and Saving
####################################

def print_online_results(results_df: pd.DataFrame, dataset_name: str):
    """
    Aggregate and print evaluation results, then save to CSV.
    
    Args:
        results_df: DataFrame with columns [Model, alpha, beta, metrics...]
        dataset_name: Name for the output CSV file
    """
    print('results_df', results_df)
    
    # Get unique model names and metric columns
    model_names = results_df['Model'].unique()
    metric_columns = [col for col in results_df.columns if col not in ['Model', 'alpha', 'beta']]
    
    # Build aggregated table
    aggregated_rows = []
    for model_name in model_names:
        model_data = results_df[results_df['Model'] == model_name]
        
        # Aggregate row with formatted statistics
        row = {
            'Model': model_name,
            'alpha': _format_parameter(model_data['alpha'].values),
            'beta': _format_parameter(model_data['beta'].values),
        }
        
        # Add formatted metrics
        for metric in metric_columns:
            row[metric] = _format_metric(model_data[metric].values)
        
        aggregated_rows.append(row)
    
    # Create and display final table
    final_table = pd.DataFrame(aggregated_rows)
    print(final_table)
    
    # Save to CSV
    csv_path = Path('real_world_datasets/results_csv') / f'{dataset_name}.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_table.to_csv(csv_path, index=False)
    print(f'\nResults saved to: {csv_path}')


def _format_metric(values):
    """Format metric values as: mean% (± std%)"""
    mean_val = np.mean(values) * 100
    std_val = np.std(values) * 100 / np.sqrt(len(values))
    return f"{mean_val:.3f} (± {std_val:.3f})"


def _format_parameter(values):
    """Format parameter values (alpha/beta) as: mean (± std) or '--' if zero"""
    mean_val = np.mean(values)
    if mean_val == 0:
        return '--'
    std_val = np.std(values) * 100 / np.sqrt(len(values))
    return f"{mean_val:.3f}(± {std_val:.3f})"
