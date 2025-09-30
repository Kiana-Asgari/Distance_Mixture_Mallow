"""
CLI package for the Distance Mixture Mallows project.

This package provides command-line interface functionality including argument parsing
and experiment execution handlers.
"""

from .parser import parse_arguments
from .handlers import run_experiment, run_synthetic_experiment, run_real_world_experiment

__all__ = [
    'parse_arguments',
    'run_experiment', 
    'run_synthetic_experiment',
    'run_real_world_experiment'
]
