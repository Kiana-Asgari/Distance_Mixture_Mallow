"""
Main script for running Distance Mixture Mallows experiments.

This script provides a command-line interface for running both synthetic and
real-world dataset experiments.
"""

import numpy as np
import sys
from cli import parse_arguments, run_experiment
from real_world_datasets.movie_lens.load_MovieLens import load_and_return_ratings_movies


if __name__ == "__main__":

    args = parse_arguments()
    run_experiment(args)