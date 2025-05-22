import numpy as np
import argparse
from real_world_datasets.fit_models import fit_models
from real_world_datasets.print_evaluations import read_and_print_results


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run preference learning models on real-world datasets.'
    )
    
    # Define subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Fit models subparser
    fit_parser = subparsers.add_parser('fit', help='Fit models to dataset')
    fit_parser.add_argument('--dataset', type=str, default='sushi', 
                           help='Dataset name (default: sushi)')
    fit_parser.add_argument('--n-teams', type=int, default=100,
                           help='Number of teams (default: 100)')
    fit_parser.add_argument('--delta', type=float, default=7,
                           help='Delta parameter (default: 7)')
    fit_parser.add_argument('--mc-samples', type=int, default=100,
                           help='Number of Monte Carlo samples (default: 100)')
    fit_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed (default: 42)')
    fit_parser.add_argument('--n-trials', type=int, default=1,
                           help='Number of trials (default: 1)')
    fit_parser.add_argument('--save', action='store_true', default=False,
                           help='Save results (default: False)')
    fit_parser.add_argument('--verbose', action='store_true', default=True,
                           help='Verbose output (default: True)')
    
    # Evaluation subparser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate and print results')
    eval_parser.add_argument('--dataset', type=str, default='football',
                            help='Dataset name (default: football)')
    eval_parser.add_argument('--n-items', type=int, default=100,
                            help='Number of items (default: 100)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.mode == 'fit':
        fit_models(dataset_name=args.dataset,
                   n_teams=args.n_teams,
                   Delta=args.delta,
                   mc_samples=args.mc_samples,
                   seed=args.seed,
                   n_trials=args.n_trials,
                   save=args.save,
                   verbose=args.verbose)
    
    elif args.mode == 'evaluate':
        read_and_print_results(n_items=args.n_items, dataset_name=args.dataset)
    
    else:
        # If no arguments provided, show help
        print("Please specify a mode. Use --help for more information.")