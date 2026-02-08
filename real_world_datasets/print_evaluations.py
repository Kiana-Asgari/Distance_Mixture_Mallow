import json
from real_world_datasets.utils import make_table
import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate


# Metric display configuration
METRIC_NICE_NAMES = {
    'spearman_rho': ('↑ Spearman\'s ρ', 100),
    'kendall_tau': ('↑ Kendall\'s τ', 100),
    'hamming_distance': ('↓ Hamming distance', 100),
    'ndcg': ('↑ nDCG', 100),
    'pairwise_acc': ('↑ Pairwise acc. (%)', 100),
}


def print_online_results(results_df: pd.DataFrame, dataset_name: str):
    """Build and print results table with smooth borders."""
    
    # Get all model names
    print('results_df', results_df)
    all_model_names = results_df['Model'].unique()
    
    # Build table rows
    rows = []
    
    # α and β parameters (only for models that use them)
    for param in ('alpha', 'beta'):
        row = [f'Estimated {param}']
        for model_name in all_model_names:
            values = _fetch(results_df, model_name, param)
            row.append(_col_param(values))
        rows.append(row)
    
    # Get metric columns (exclude Model, alpha, beta)
    metric_columns = [col for col in results_df.columns 
                     if col not in ['Model', 'alpha', 'beta']]
    
    # Add metrics (with nice names if available)
    for metric in metric_columns:
        # Check if we have special handling for top_k_hit_rates
        if metric == 'top_k_hit_rates':
            # Handle top-k metrics (k=1 and k=5)
            for k in (1, 5):
                row = [f'↑ Top-{k} hit rate (%)']
                for model_name in all_model_names:
                    values = _fetch_top_k(results_df, model_name, k)
                    row.append(_col(values, scale=100))
                rows.append(row)
        else:
            # Regular metric
            label, scale = METRIC_NICE_NAMES.get(metric, (metric, 100))
            row = [label]
            for model_name in all_model_names:
                values = _fetch(results_df, model_name, metric)
                row.append(_col(values, scale=scale))
            rows.append(row)
    
    # Print table with smooth borders
    print('\n' + '='*80)
    print(f'Results for: {dataset_name}')
    print('='*80)
    print(tabulate(rows, headers=['Metric'] + list(all_model_names),
                   tablefmt='fancy_grid'))
    print('='*80 + '\n')
    
    # Save results to CSV (keep original format)
    _save_results_csv(results_df, all_model_names, metric_columns, dataset_name)




####################################
# Helper functions
####################################

def _fetch(results_df: pd.DataFrame, model_name: str, column: str):
    """Extract values for a given model and column."""
    model_rows = results_df[results_df['Model'] == model_name]
    return model_rows[column].values


def _fetch_top_k(results_df: pd.DataFrame, model_name: str, k: int):
    """Extract top-k hit rate values (k-th element from each list)."""
    model_rows = results_df[results_df['Model'] == model_name]
    top_k_lists = model_rows['top_k_hit_rates'].values
    # Extract k-th element (k-1 index) from each list
    return np.array([lst[k-1] if isinstance(lst, (list, np.ndarray)) and len(lst) >= k 
                     else 0.0 for lst in top_k_lists])


def _col(values, scale=1):
    """Format column as: mean (± std), optionally scaled (e.g., to %)."""
    mean_val = np.mean(values) * scale
    std_val = np.std(values) * scale
    return f"{mean_val:.3f} (± {std_val:.3f})"


def _col_param(values):
    """Format parameter column (alpha/beta), returning '--' if all zeros."""
    if np.all(values == 0):
        return '--'
    mean_val = np.mean(values)
    std_val = np.std(values)
    return f"{mean_val:.3f} (± {std_val:.3f})"


def _save_results_csv(results_df, model_names, metric_columns, dataset_name):
    """Save aggregated results to CSV."""
    csv_rows = []
    
    for model_name in model_names:
        row = {'Model': model_name}
        
        # Add parameters
        for param in ('alpha', 'beta'):
            values = _fetch(results_df, model_name, param)
            row[param] = _col_param(values)
        
        # Add metrics
        for metric in metric_columns:
            if metric == 'top_k_hit_rates':
                # Add top-1 and top-5 separately
                for k in (1, 5):
                    values = _fetch_top_k(results_df, model_name, k)
                    row[f'top_{k}_hit_rate'] = _col(values, scale=100)
            else:
                values = _fetch(results_df, model_name, metric)
                scale = METRIC_NICE_NAMES.get(metric, (metric, 100))[1]
                row[metric] = _col(values, scale=scale)
        
        csv_rows.append(row)
    
    # Save to CSV
    csv_df = pd.DataFrame(csv_rows)
    csv_dir = Path('real_world_datasets/results_csv')
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f'{dataset_name}.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f'Results saved to: {csv_path}')


def read_and_print_results(n_items=100, dataset_name='basketball'):
    """Read and print results from JSON file."""
    results_file = f'real_world_datasets/results/{dataset_name}_n_teams={n_items}.json'
    print(f'\nThe results for {n_items} teams of {dataset_name} are:\n')
    # Load the results
    with open(results_file, 'r') as f:
        results = json.load(f)
    make_table(results)




