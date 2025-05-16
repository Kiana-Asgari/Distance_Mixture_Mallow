import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def read_model_comparisons_basketball(n_top_teams=10,
                          output_dir='basketball/results/plots'):
    results_file=f'basketball/results/basketball_2019_n_top_teams={n_top_teams}(chronological).json'

        # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} trials for plotting")

    alpha = {
        'Original L1': [trial['alpha'] for trial in results],    }
    beta = {
        'Original L1': [trial['beta'] for trial in results],
    }
    
    # Extract NDCG metrics for each trial and model
    ndcg_scores = {
        'Original L1': [trial['ndcg_ML'] for trial in results],
        'Plackett-Luce': [trial['ndcg_PL'] for trial in results],
        'Kendall': [trial['ndcg_kendal'] for trial in results],
    }
    top_k_hit_rates = {
        'Original L1': [trial['top_k_hit_rates_ML'] for trial in results],
        'Plackett-Luce': [trial['top_k_hit_rates_PL'] for trial in results],
        'Kendall': [trial['top_k_hit_rates_kendal'] for trial in results],
    }
    spearman_rho = {
        'Original L1': [trial['spearman_rho_ML'] for trial in results],
        'Plackett-Luce': [trial['spearman_rho_PL'] for trial in results],
        'Kendall': [trial['spearman_rho_kendal'] for trial in results],
    }
    kendall_tau = {
        'Original L1': [trial['kendall_tau_ML'] for trial in results],
        'Plackett-Luce': [trial['kendall_tau_PL'] for trial in results],
        'Kendall': [trial['kendall_tau_kendal'] for trial in results],
    }
    hamming_distance = {
        'Original L1': [trial['hamming_distance_ML'] for trial in results],
        'Plackett-Luce': [trial['hamming_distance_PL'] for trial in results],
        'Kendall': [trial['hamming_distance_kendal'] for trial in results],
    }

    pairwise_acc = {
        'Original L1': [trial['pairwise_acc_ML'] for trial in results],
        'Plackett-Luce': [trial['pairwise_acc_PL'] for trial in results],
        'Kendall': [trial['pairwise_acc_kendal'] for trial in results],
    }
   
    print('========Table of results for Original L1========')
    print('mean and std of alpha: ', np.mean(alpha['Original L1']), np.std(alpha['Original L1']))
    print('mean and std of beta: ', np.mean(beta['Original L1']), np.std(beta['Original L1']))
    print('mean and std of originalndcg: ', np.mean(ndcg_scores['Original L1']), np.std(ndcg_scores['Original L1']))
    print('mean and std of top_1_hit_rates: ', np.mean(top_k_hit_rates['Original L1'], axis=0)[0], np.std(top_k_hit_rates['Original L1'], axis=0)[0])
    print('mean and std of top_5_hit_rates: ', np.mean(top_k_hit_rates['Original L1'], axis=0)[4], np.std(top_k_hit_rates['Original L1'], axis=0)[4])
    print('mean and std of spearman_rho: ', np.mean(spearman_rho['Original L1']), np.std(spearman_rho['Original L1']))
    print('mean and std of kendall_tau: ', np.mean(kendall_tau['Original L1']), np.std(kendall_tau['Original L1']))
    print('mean and std of hamming_distance: ', np.mean(hamming_distance['Original L1']), np.std(hamming_distance['Original L1']))
    print('mean and std of pairwise_acc: ', np.mean(pairwise_acc['Original L1']), np.std(pairwise_acc['Original L1']))


    print('========Table of results for Plackett-Luce========')
    print('mean and std of ndcg: ', np.mean(ndcg_scores['Plackett-Luce']), np.std(ndcg_scores['Plackett-Luce']))
    print('mean and std of top_1_hit_rates: ', np.mean(top_k_hit_rates['Plackett-Luce'], axis=0)[0], np.std(top_k_hit_rates['Plackett-Luce'], axis=0)[0])
    print('mean and std of top_5_hit_rates: ', np.mean(top_k_hit_rates['Plackett-Luce'], axis=0)[4], np.std(top_k_hit_rates['Plackett-Luce'], axis=0)[4])
    print('mean and std of spearman_rho: ', np.mean(spearman_rho['Plackett-Luce']), np.std(spearman_rho['Plackett-Luce']))
    print('mean and std of kendall_tau: ', np.mean(kendall_tau['Plackett-Luce']), np.std(kendall_tau['Plackett-Luce']))
    print('mean and std of hamming_distance: ', np.mean(hamming_distance['Plackett-Luce']), np.std(hamming_distance['Plackett-Luce']))
    print('mean and std of pairwise_acc: ', np.mean(pairwise_acc['Plackett-Luce']), np.std(pairwise_acc['Plackett-Luce']))
    
    print('========Table of results for Kendall========')
    print('mean and std of ndcg: ', np.mean(ndcg_scores['Kendall']), np.std(ndcg_scores['Kendall']))
    print('mean and std of top_1_hit_rates: ', np.mean(top_k_hit_rates['Kendall'], axis=0)[0], np.std(top_k_hit_rates['Kendall'], axis=0)[0])
    print('mean and std of top_5_hit_rates: ', np.mean(top_k_hit_rates['Kendall'], axis=0)[4], np.std(top_k_hit_rates['Kendall'], axis=0)[4])
    print('mean and std of spearman_rho: ', np.mean(spearman_rho['Kendall']), np.std(spearman_rho['Kendall']))
    print('mean and std of kendall_tau: ', np.mean(kendall_tau['Kendall']), np.std(kendall_tau['Kendall']))
    print('mean and std of hamming_distance: ', np.mean(hamming_distance['Kendall']), np.std(hamming_distance['Kendall']))
    print('mean and std of pairwise_acc: ', np.mean(pairwise_acc['Kendall']), np.std(pairwise_acc['Kendall']))


   