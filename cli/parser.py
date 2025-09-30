"""
Command-line argument parsing for the Distance Mixture Mallows project.

This module provides clean, modular argument parsing for different experiment modes.
"""

import argparse


def add_common_arguments(parser):
    """Add common arguments shared across subparsers."""
    parser.add_argument('--n-trials', type=int, default=1,
                       help='Number of trials (default: 1)')
    parser.add_argument('--save', action='store_true', default=False,
                       help='Save results (default: False)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output (default: True)')
    return parser


def add_real_world_arguments(parser):
    """Add arguments specific to real-world dataset fitting."""
    parser.add_argument('--dataset', type=str, default='movie_lens',
                       choices=['movie_lens', 'sushi', 'news', 'basketball', 'football'],
                       help='Dataset name (default: movie_lens)')
    parser.add_argument('--n-teams', type=int, default=100,
                       help='Number of teams (default: 100)')
    parser.add_argument('--truncation', type=int, choices=[5, 6, 7], default=7,
                       help='Truncation parameter (default: 7, choices: 5-7)')
    parser.add_argument('--mc-samples', type=int, default=500,
                       help='Number of Monte Carlo samples (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--n-movies', type=int, default=10,
                       help='Number of movies (default: 10)')
    return parser


def add_synthetic_arguments(parser):
    """Add arguments specific to synthetic data fitting."""
    parser.add_argument('--n-items', type=int, default=15, choices=[10, 15, 20],
                       help='Number of items (default: 15, choices: 10,15,20)')
    parser.add_argument('--alpha-0', type=float, default=1.0,
                       help='Alpha_0 parameter (default: 1.0)')
    parser.add_argument('--beta-0', type=float, default=1.0,
                       help='Beta_0 parameter (default: 1.0)')
    parser.add_argument('--n-train', type=int, nargs='+', default=[10, 50, 200],
                       help='Number of training samples (default: [10,50,200])')
    parser.add_argument('--truncation-training', type=int, choices=[3, 4, 5, 6], default=6,
                       help='Truncation parameter (default: 6, choices: 3-6)')
    parser.add_argument('--truncation-data-generation', type=int, default=8,
                       help='Truncation parameter for data generation (default: 8)')
    return parser


def parse_arguments():
    """Parse command line arguments with clean separation of concerns."""
    parser = argparse.ArgumentParser(
        description='Run preference learning models on real-world datasets.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Define subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)
    
    # Real-world dataset subparser
    real_world_parser = subparsers.add_parser('fit-real-world', 
                                            help='Fit models to real-world dataset')
    real_world_parser = add_common_arguments(real_world_parser)
    real_world_parser = add_real_world_arguments(real_world_parser)
    
    # Synthetic data subparser
    synthetic_parser = subparsers.add_parser('fit-synthetic', 
                                           help='Fit models to synthetic data')
    synthetic_parser = add_common_arguments(synthetic_parser)
    synthetic_parser = add_synthetic_arguments(synthetic_parser)
    
    return parser.parse_args()
