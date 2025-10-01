import json
from real_world_datasets.utils import make_table
import pandas as pd
import numpy as np

def print_online_results(results: dict):
    print(results)

    all_model_names = list(results.keys())
    all_metric_names = list(results[all_model_names[0]][0]['evals'].keys())
    columns = ['Model', 'alpha', 'beta'] + list(all_metric_names)
    final_table = pd.DataFrame(columns=columns)

    for model_name in all_model_names:
        model_results = results[model_name]
        
        # Extract all values for this model
        alpha_values = [trial['args'].get('alpha', 0) for trial in model_results]
        beta_values = [trial['args'].get('beta', 0) for trial in model_results]
        metric_values = {metric: [trial['evals'][metric] for trial in model_results] 
                        for metric in all_metric_names}
        
        
        # Create row data
        row_data = {
            'Model': model_name,
            'alpha': _format_param(alpha_values, is_arg=True),
            'beta': _format_param(beta_values, is_arg=True),
            **{metric: _format_stat(values, is_arg=False) for metric, values in metric_values.items()}
        }
        
        final_table = pd.concat([final_table, pd.DataFrame([row_data])], ignore_index=True)

    print(final_table)



def read_and_print_results(n_items=100, dataset_name='basketball'):
         
    results_file = f'real_world_datasets/results/{dataset_name}_n_teams={n_items}.json'
    print(f' \n The resuls for {n_items} teams of {dataset_name} are:\n')
    # Load the results
    with open(results_file, 'r') as f:
        results = json.load(f)
    make_table(results)


# Calculate statistics
def _format_stat(values, is_arg: bool = False):
    if is_arg:
        mean_val = np.mean(values)
        se_val = np.std(values) / np.sqrt(len(values))
        return f"{mean_val:.2f}(± {se_val:.1f})"
    else:
        mean_val = np.mean(values) * 100
        se_val = np.std(values) / np.sqrt(len(values)) * 100
        return f"{mean_val:.0f} (± {se_val:.0f})"

def _format_param(values, is_arg: bool = False):
    mean_val = np.mean(values)
    return _format_stat(values, is_arg) if mean_val != 0 else '--'




