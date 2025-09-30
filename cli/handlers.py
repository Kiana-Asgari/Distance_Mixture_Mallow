"""
Experiment execution handlers for the Distance Mixture Mallows project.

This module contains functions that handle the execution of different experiment types.
"""

from real_world_datasets.fit_models import fit_models
from synthethic_tests.synthethic_script import learn_synthetic_data


def get_dataset_config(args):
    """Get dataset-specific configuration."""
    config = {
        'n_teams': args.n_teams,
        'dataset_name': args.dataset
    }
    
    if args.dataset == 'sushi':
        config['n_teams'] = 10
    elif args.dataset == 'movie_lens':
        config['n_teams'] = args.n_movies
    
    return config


def run_synthetic_experiment(args):
    """Run synthetic data experiment."""
    learn_synthetic_data(
        n=args.n_items,
        Delta=args.truncation_training,
        Delta_data=args.truncation_data_generation,
        beta_0=args.beta_0,
        alpha_0=args.alpha_0,
        save=args.save,
        num_train_samples=args.n_train,
        n_trials=args.n_trials,
        verbose=args.verbose
    )


def run_real_world_experiment(args):
    """Run real-world dataset experiment."""
    config = get_dataset_config(args)
    
    fit_models(
        dataset_name=config['dataset_name'],
        n_teams=config['n_teams'],
        Delta=args.truncation,
        mc_samples=args.mc_samples,
        seed=args.seed,
        n_trials=args.n_trials,
        save=args.save,
        verbose=args.verbose
    )


def run_experiment(args):
    """Main experiment runner that dispatches to appropriate handler."""
    if args.mode == 'fit-synthetic':
        run_synthetic_experiment(args)
    elif args.mode == 'fit-real-world':
        run_real_world_experiment(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
