import json
from real_world_datasets.utils import make_table
import pandas as pd
import numpy as np
from pathlib import Path

def print_online_results(results_df: pd.DataFrame, dataset_name: str ):
    
    # Get all model names
    print('results_df', results_df)
    all_model_names = results_df['Model'].unique()
    
    # Get metric column names (all columns except Model, alpha, beta)
    metric_columns = [col for col in results_df.columns if col not in ['Model', 'alpha', 'beta']]
    
    # Create final table with same structure
    columns = ['Model', 'alpha', 'beta'] + metric_columns
    final_table = pd.DataFrame(columns=columns)

    for model_name in all_model_names:
        # Filter rows for this model
        model_df = results_df[results_df['Model'] == model_name]
        
        # Extract all values for this model
        alpha_values = model_df['alpha'].values
        beta_values = model_df['beta'].values
        
        # Aggregate metrics: compute mean and std for each metric
        aggregated_row = {
            'Model': model_name,
            'alpha': _format_param(alpha_values, is_arg=True),
            'beta': _format_param(beta_values, is_arg=True)
        }
        
        # Add formatted statistics for each metric
        for metric in metric_columns:
            metric_values = model_df[metric].values
            aggregated_row[metric] = _format_stat(metric_values, is_arg=False)
        
        # Add row to final table
        final_table.loc[len(final_table)] = aggregated_row

    print(final_table)
    
    # Save results to CSV
    csv_dir = Path('real_world_datasets/results_csv')
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f'{dataset_name}.csv'
    final_table.to_csv(csv_path, index=False)
    print(f'\nResults saved to: {csv_path}')




def read_and_print_results(n_items=100, dataset_name='basketball'):
         
    results_file = f'real_world_datasets/results/{dataset_name}_n_teams={n_items}.json'
    print(f' \n The resuls for {n_items} teams of {dataset_name} are:\n')
    # Load the results
    with open(results_file, 'r') as f:
        results = json.load(f)
    make_table(results)


# Calculate statistics
def _format_stat(values, is_arg: bool = False):
    if is_arg and '--' not in values:
        mean_val = np.mean(values)
        se_val = np.std(values) * 100 / np.sqrt(len(values))
        return f"{mean_val:.3f}(± {se_val:.3f})"
    elif is_arg and '--' in values:
        return '--'
    else:
        mean_val = np.mean(values) * 100
        se_val = np.std(values) * 100 / np.sqrt(len(values))
        return f"{mean_val:.3f} (± {se_val:.3f})"

def _format_param(values, is_arg: bool = False):
    mean_val = np.mean(values)
    return _format_stat(values, is_arg) if mean_val != 0 else '--'




