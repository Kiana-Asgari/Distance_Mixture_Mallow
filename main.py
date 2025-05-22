import numpy as np
from real_world_datasets.college_sports.fit_models import fit_models
from real_world_datasets.print_evaluations import read_and_print_results



if __name__ == "__main__":
    fit_models(dataset_name="basketball",
               n_teams=100,
               Delta=7,
               mc_samples=100,
               seed=42,
               n_trials=2,
               save=False,
               verbose=True)
    #read_and_print_results(n_items=10, dataset_name='football')