import json
from plotting_utils import make_table


def read_model_comparisons_basketball(n_top_teams=100):
         
    results_file = f'basketball/results/basketball_2019_n_top_teams={n_top_teams}(chronological).json'
    print(f' \n The resuls for {n_top_teams} teams of basketball are:\n')
    # Load the results
    with open(results_file, 'r') as f:
        results = json.load(f)
    make_table(results)





