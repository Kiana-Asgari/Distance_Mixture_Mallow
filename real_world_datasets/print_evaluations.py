import json
from real_world_datasets.utils import make_table


def print_online_results(results):
    make_table(results)


def read_and_print_results(n_items=100, dataset_name='basketball'):
         
    results_file = f'real_world_datasets/results/{dataset_name}_n_teams={n_items}.json'
    print(f' \n The resuls for {n_items} teams of {dataset_name} are:\n')
    # Load the results
    with open(results_file, 'r') as f:
        results = json.load(f)
    make_table(results)





