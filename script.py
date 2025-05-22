import numpy as np
import argparse
from real_world_datasets.fit_models import fit_models
from real_world_datasets.print_evaluations import read_and_print_results
from synthethic_tests.synthethic_script import learn_synthetic_data
from synthethic_tests.synthethic_script import test_effect_of_truncation,test_effect_of_n

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run preference learning models on real-world datasets.'
    )
    
    # Define subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Fit models subparser
    fit_real_world_parser = subparsers.add_parser('fit-real-world', help='Fit models to dataset')
    fit_real_world_parser.add_argument('--dataset', type=str, default='sushi', 
                           help='Dataset name (default: sushi)')
    fit_real_world_parser.add_argument('--n-teams', type=int, default=100,
                           help='Number of teams (default: 100)')
    fit_real_world_parser.add_argument('--truncation', type=int, choices=[5,6,7], default=7,
                           help='Truncation parameter (default: 7, choices: 5-7)')
    fit_real_world_parser.add_argument('--mc-samples', type=int, default=100,
                           help='Number of Monte Carlo samples (default: 100)')
    fit_real_world_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed (default: 42)')
    fit_real_world_parser.add_argument('--n-trials', type=int, default=1,
                           help='Number of trials (default: 1)')
    fit_real_world_parser.add_argument('--save', action='store_true', default=False,
                           help='Save results (default: False)')
    fit_real_world_parser.add_argument('--verbose', action='store_true', default=True,
                           help='Verbose output (default: True)')
    
    # Synthetic data subparser
    fit_synthetic_parser = subparsers.add_parser('fit-synthetic', help='Fit models to synthetic data')    
    fit_synthetic_parser.add_argument('--n-items', type=int, default=15, choices=[10, 15, 20],
                           help='Number of items (default: 15, choices: 10,15,20)')
    fit_synthetic_parser.add_argument('--alpha-0', type=float, default=1,
                        help='Alpha_0 parameter (default: 1)')
    fit_synthetic_parser.add_argument('--beta-0', type=float, default=1,
                        help='Beta_0 parameter (default:1)')
    fit_synthetic_parser.add_argument('--n_train', type=int, nargs='+', default=[10, 50, 200],
                        help='Number of training samples (default: [10,50,200])')
    fit_synthetic_parser.add_argument('--n-trials', type=int, default=1,
                           help='Number of trials (default: 1)')
    fit_synthetic_parser.add_argument('--truncation_training', type=int, choices=[3,4,5,6], default=6,
                           help='Truncation parameter (default: 6, choices: 3-6)')
    fit_synthetic_parser.add_argument('--save', action='store_true', default=False,
                           help='Save results (default: False)')
    fit_synthetic_parser.add_argument('--verbose', action='store_true', default=True,
                           help='Verbose output (default: True)')
    fit_synthetic_parser.add_argument('--truncation_data_generation', type=int, default=8,
                           help='Truncation parameter for data generation (default: 8)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.mode == 'fit-synthetic':
        learn_synthetic_data(n=args.n_items,
                             Delta=args.truncation_training,
                             Delta_data = args.truncation_data_generation,
                             beta_0=args.beta_0,
                             alpha_0=args.alpha_0,
                             save=args.save,
                             num_train_samples=args.n_train,
                             n_trials=args.n_trials,
                             verbose=args.verbose)

    
    elif args.mode == 'fit-real-world':
        if args.dataset == 'sushi':
            args.n_teams = 10

        fit_models(dataset_name=args.dataset,
                   n_teams=args.n_teams,
                   Delta=args.truncation,
                   mc_samples=args.mc_samples,
                   seed=args.seed,
                   n_trials=args.n_trials,
                   save=args.save,
                   verbose=args.verbose)
    
    else:
        # If no arguments provided, show help
        print("Please specify a mode. Use --help for more information.")