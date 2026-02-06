"""
Main script for running Distance Mixture Mallows experiments.

This script provides a command-line interface for running both synthetic and
real-world dataset experiments.
"""

from cli import parse_arguments, run_experiment


if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(args)