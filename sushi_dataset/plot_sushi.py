import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def read_model_comparisons_sushi(results_file='sushi_dataset/results/sushi_fit_results.json', 
                          output_dir='sushi_dataset/results/plots'):

    """
    Creates box plots comparing the NDCG scores of the three models 
    (original, PL, and Kendall) across all trials.
    
    Parameters
    ----------
    results_file : str
        Path to the JSON file containing the trial results
    output_dir : str
        Directory where to save the plot image
    """
    results_file_L0 = 'sushi_dataset/results/sushi_fit_results(L0).json'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the results
    with open(results_file, 'r') as f:
        results = json.load(f)
    with open(results_file_L0, 'r') as f:
        results_L0 = json.load(f)
    print(f"Loaded {len(results)} trials for plotting")

    alpha = {
        'Original': [trial['alpha'] for trial in results],
        'Original L0': [trial['alpha'] for trial in results_L0],
    }
    beta = {
        'Original': [trial['beta'] for trial in results],
        'Original L0': [trial['beta'] for trial in results_L0],
    }
    
    # Extract NDCG metrics for each trial and model
    ndcg_scores = {
        'Original': [trial['ndcg_ML'] for trial in results],
        'Plackett-Luce': [trial['ndcg_PL'] for trial in results],
        'Kendall': [trial['ndcg_kendal'] for trial in results],
        'Spearman': [trial['ndcg_spearman'] for trial in results],
        'Original L0': [trial['ndcg_ML'] for trial in results_L0],
    }
    top_k_hit_rates = {
        'Original': [trial['top_k_hit_rates_ML'] for trial in results],
        'Plackett-Luce': [trial['top_k_hit_rates_PL'] for trial in results],
        'Kendall': [trial['top_k_hit_rates_kendal'] for trial in results],
        'Spearman': [trial['top_k_hit_rates_spearman'] for trial in results],
        'Original L0': [trial['top_k_hit_rates_ML'] for trial in results_L0],
    }
    spearman_rho = {
        'Original': [trial['spearman_rho_ML'] for trial in results],
        'Plackett-Luce': [trial['spearman_rho_PL'] for trial in results],
        'Kendall': [trial['spearman_rho_kendal'] for trial in results],
        'Spearman': [trial['spearman_rho_spearman'] for trial in results],
        'Original L0': [trial['spearman_rho_ML'] for trial in results_L0],
    }
    kendall_tau = {
        'Original': [trial['kendall_tau_ML'] for trial in results],
        'Plackett-Luce': [trial['kendall_tau_PL'] for trial in results],
        'Kendall': [trial['kendall_tau_kendal'] for trial in results],
        'Spearman': [trial['kendall_tau_spearman'] for trial in results],
        'Original L0': [trial['kendall_tau_ML'] for trial in results_L0],
    }
    hamming_distance = {
        'Original': [trial['hamming_distance_ML'] for trial in results],
        'Plackett-Luce': [trial['hamming_distance_PL'] for trial in results],
        'Kendall': [trial['hamming_distance_kendal'] for trial in results],
        'Spearman': [trial['hamming_distance_spearman'] for trial in results],
        'Original L0': [trial['hamming_distance_ML'] for trial in results_L0],
    }

    pairwise_acc = {
        'Original': [trial['pairwise_acc_ML'] for trial in results],
        'Plackett-Luce': [trial['pairwise_acc_PL'] for trial in results],
        'Kendall': [trial['pairwise_acc_kendal'] for trial in results],
        'Spearman': [trial['pairwise_acc_spearman'] for trial in results],
        'Original L0': [trial['pairwise_acc_ML'] for trial in results_L0],
    }
    print('========Table of results L1========')
    print('mean and std of alpha: ', np.mean(alpha['Original']), np.std(alpha['Original']))
    print('mean and std of beta: ', np.mean(beta['Original']), np.std(beta['Original']))
    print('mean and std of originalndcg: ', np.mean(ndcg_scores['Original']), np.std(ndcg_scores['Original']))
    print('mean and std of top_1_hit_rates: ', np.mean(top_k_hit_rates['Original'], axis=0)[0], np.std(top_k_hit_rates['Original'], axis=0)[0])
    print('mean and std of top_5_hit_rates: ', np.mean(top_k_hit_rates['Original'], axis=0)[4], np.std(top_k_hit_rates['Original'], axis=0)[4])
    print('mean and std of spearman_rho: ', np.mean(spearman_rho['Original']), np.std(spearman_rho['Original']))
    print('mean and std of kendall_tau: ', np.mean(kendall_tau['Original']), np.std(kendall_tau['Original']))
    print('mean and std of hamming_distance: ', np.mean(hamming_distance['Original']), np.std(hamming_distance['Original']))
    print('mean and std of pairwise_acc: ', np.mean(pairwise_acc['Original']), np.std(pairwise_acc['Original']))

    print('========Table of results for Original L0========')
    print('mean and std of alpha: ', np.mean(alpha['Original L0']), np.std(alpha['Original L0']))
    print('mean and std of beta: ', np.mean(beta['Original L0']), np.std(beta['Original L0']))
    print('mean and std of originalndcg: ', np.mean(ndcg_scores['Original L0']), np.std(ndcg_scores['Original L0']))
    print('mean and std of top_1_hit_rates: ', np.mean(top_k_hit_rates['Original L0'], axis=0)[0], np.std(top_k_hit_rates['Original L0'], axis=0)[0])
    print('mean and std of top_5_hit_rates: ', np.mean(top_k_hit_rates['Original L0'], axis=0)[4], np.std(top_k_hit_rates['Original L0'], axis=0)[4])
    print('mean and std of spearman_rho: ', np.mean(spearman_rho['Original L0']), np.std(spearman_rho['Original L0']))
    print('mean and std of kendall_tau: ', np.mean(kendall_tau['Original L0']), np.std(kendall_tau['Original L0']))
    print('mean and std of hamming_distance: ', np.mean(hamming_distance['Original L0']), np.std(hamming_distance['Original L0']))
    print('mean and std of pairwise_acc: ', np.mean(pairwise_acc['Original L0']), np.std(pairwise_acc['Original L0']))
    
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


    print('========Table of results for Spearman========')
    print('mean and std of ndcg: ', np.mean(ndcg_scores['Spearman']), np.std(ndcg_scores['Spearman']))
    print('mean and std of top_1_hit_rates: ', np.mean(top_k_hit_rates['Spearman'], axis=0)[0], np.std(top_k_hit_rates['Spearman'], axis=0)[0])
    print('mean and std of top_5_hit_rates: ', np.mean(top_k_hit_rates['Spearman'], axis=0)[4], np.std(top_k_hit_rates['Spearman'], axis=0)[4])
    print('mean and std of spearman_rho: ', np.mean(spearman_rho['Spearman']), np.std(spearman_rho['Spearman']))
    print('mean and std of kendall_tau: ', np.mean(kendall_tau['Spearman']), np.std(kendall_tau['Spearman']))
    print('mean and std of hamming_distance: ', np.mean(hamming_distance['Spearman']), np.std(hamming_distance['Spearman']))
    print('mean and std of pairwise_acc: ', np.mean(pairwise_acc['Spearman']), np.std(pairwise_acc['Spearman']))


