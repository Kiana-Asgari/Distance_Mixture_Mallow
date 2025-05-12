import numpy as np
import sys
import time

from GMM_diagonalized.sampling import sample_truncated_mallow



if __name__ == "__main__":
    print('****************************Running main.py****************************')


    seq_results = sample_truncated_mallow(num_samples=1, n=30, beta=0.5, alpha=1.0, Delta=6)



