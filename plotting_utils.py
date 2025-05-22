from matplotlib import pyplot as plt
import numpy as np
from GMM_diagonalized.sampling import sample_truncated_mallow
from tabulate import tabulate







####################################
# Plotting utils
####################################
def set_publication_style():
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })


def plot_marginal_heamap(n, alpha, beta, Delta):

    sigma = 1+np.arange(n)
    sampled_perm = sample_truncated_mallow(num_samples=10_000,
                                          n=n, 
                                          alpha=alpha, 
                                          beta=beta, 
                                          sigma=sigma,
                                          Delta=Delta)
    marginal_counts = np.zeros((n, n))
    for perm in sampled_perm:
        for i in range(n):
            marginal_counts[i, perm[i] - 1] += 1  # Subtract 1 to convert 1-indexed to 0-indexed
    
    # Normalize to get probabilities
    marginal_counts /= len(sampled_perm)
    set_publication_style()
    
    # Create figure with appropriate size
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap with a probability-appropriate colormap
    im = plt.imshow(marginal_counts, cmap='viridis', vmin=0, vmax=0.6)
    
    # Add colorbar with label
    cbar = plt.colorbar(im)
    cbar.set_label('Probability', rotation=270, labelpad=20)
    
    # Add labels and title
    #plt.xlabel('Item $j$')
    #plt.ylabel('Position $i$')
    #plt.title(r'Marginal Probabilities P($\pi$(i)=j) for Mallows Model' + f'\n($\\alpha$={alpha}, $\\beta$={beta}, $\\Delta$={Delta})')
    
    # Set ticks to show 1-indexed values
    plt.xticks(np.arange(n), np.arange(1, n+1))
    plt.yticks(np.arange(n), np.arange(1, n+1))
    print(np.sum(marginal_counts, axis=0))
    
    plt.tight_layout()
    plt.savefig(f"marginal_counts_{n}_{alpha}_{beta}_{Delta}.pdf", bbox_inches='tight')




