import numpy as np
import sys
import time

from GMM_diagonalized.sampling import sample_truncated_mallow
from MLE.consensus_ranking_estimation import consensus_ranking_estimation
from MLE.alpha_beta_estimation import solve_alpha_beta
from synthethic_tests.synthethic_script import save_synthetic_data
from synthethic_tests.plot import plot_alpha_vs_n_samples, plot_beta_vs_n_samples
from sushi_dataset.load_data import load_sushi
from sushi_dataset.fit_sushi import fit_and_save_sushi
from basketball.plot_bascketball import plot_model_comparisons_basketball
from football.plot_football import read_model_comparisons_football
from sushi_dataset.plots import plot_model_comparisons_sushi
from football.fit_football import fit_football
from basketball.fit_bascketball import fit_basketball
from university.fit_university import fit_uni
from APA.fit_APA import fit_apa
from plotting_utils import plot_marginal_heamap
if __name__ == "__main__":
    print('****************************Running main.py****************************')
    #fit_football(Delta=8,n_teams_to_keep=10)
    fit_and_save_sushi()
    #read_model_comparisons_football()
    #save_synthetic_data(n=10, alpha_0=1.5, beta_0=0.5, Delta=6)
    #plot_alpha_vs_n_samples(alpha_0=1.5, beta_0=0.5, n='15(main)')
   # plot_beta_vs_n_samples(alpha_0=1.5, beta_0=0.5, n='15(main)')
    #sushi_data = load_sushi()
   # print(sushi_data)
    #fit_and_save_sushi()
    # fit_football(n_file=100,n_top_teams=11,n_bottom_teams=1)
    #fit_apa()
    # fit_uni(n_file=100,n_top_teams=11,n_bottom_teams=1)
    #fit_basketball(n_file=100,n_top_teams=21,n_bottom_teams=15,Delta=7,seed=42)
    #plot_model_comparisons()
    #plot_model_comparisons_sushi()
    # sys.exit()
    # n = 10
    # Delta = 6
    # sigma_0 = 1+np.arange(n)
    # beta_0 = 0.3
    # alpha_0 = 1.5
    # num_train_samples = 1000



    # train_samples = sample_truncated_mallow(num_samples=num_train_samples,
    #                                          n=n, beta=beta_0, alpha=alpha_0,
    #                                         sigma=sigma_0, Delta=9)
    # print(f'done sampling {num_train_samples} samples')
    # consensus_ranking = consensus_ranking_estimation(train_samples)

    # print(f'done estimating consensus ranking: {consensus_ranking}')
    # params = solve_alpha_beta(train_samples, consensus_ranking)
    # print('alpha beta estimation finished', params)
