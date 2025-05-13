import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_model_comparisons_basketball(results_file='basketball/results/basketball_fit_results.json', 
                          output_dir='basketball/results/plots'):
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
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} trials for plotting")
    
    # Extract NDCG metrics for each trial and model
    ndcg_scores = {
        'Original': [trial['ndcg'] for trial in results],
        'Plackett-Luce': [trial['ndcg_PL'] for trial in results],
        'Kendall': [trial['ndcg_kendal'] for trial in results]
    }
    print("\nNDCG Statistics:")
    print("-" * 40)
    print(f"{'Model':<15} {'Mean':<10} {'Std Dev':<10}")
    print("-" * 40)
    for model, scores in ndcg_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{model:<15} {mean_score:.4f}     {std_score:.4f}")
    print("-" * 40)
    
    # Plot NDCG scores using seaborn
    data = []
    for model, scores in ndcg_scores.items():
        for trial_idx, score in enumerate(scores):
            data.append({
                'Model': model,
                'Trial': trial_idx + 1,
                'NDCG Score': score
            })
    
    # Convert to DataFrame
    ndcg_df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='Model', y='NDCG Score', data=ndcg_df)
    plt.title('NDCG Scores Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ndcg_comparison.png'), dpi=300)
    plt.close()
    
    print(f"NDCG comparison plot saved to {output_dir}")
