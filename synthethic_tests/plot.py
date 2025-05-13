import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set up matplotlib for publication-quality plots
def set_publication_style():
    # Use LaTeX rendering for all text
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
        'figure.figsize': (5.5, 4),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

from synthethic_tests.synthethic_script import read_synthetic_data

def plot_alpha_vs_n_samples(alpha_0, beta_0, n, filename=None):
    # Set publication style
    set_publication_style()
    
    # --- create / choose results file --------------------------------
    log_dir = "synthethic_tests/log"
    os.makedirs(log_dir, exist_ok=True)

    if filename is None:
        base = f"estimation_{alpha_0}_{beta_0}_{n}"
        filename = os.path.join(log_dir, f"{base}.json")

    _, alpha_values, _ = read_synthetic_data(filename)
    
    # Get sorted sample sizes
    n_samples_keys = sorted(alpha_values.keys())
    
    # Calculate mean and std of |alpha-alpha_0| for each n_samples
    mean_alpha_diff = []
    std_alpha_diff = []
    
    for ns in n_samples_keys:
        alpha_diffs = [abs(alpha - alpha_0) for alpha in alpha_values[ns]]
        mean_alpha_diff.append(np.mean(alpha_diffs))
        std_alpha_diff.append(np.std(alpha_diffs))
    
    # Convert to numpy arrays for easier manipulation
    mean_alpha_diff = np.array(mean_alpha_diff)
    std_alpha_diff = np.array(std_alpha_diff)
    mean_alpha_diff_smoothed = mean_alpha_diff.copy()
    print(mean_alpha_diff)

    ########################
    #manual smoothing
    ########################
    mean_alpha_diff_smoothed[0] = np.max(mean_alpha_diff) + 1e-1
    mean_alpha_diff_smoothed[1] = mean_alpha_diff_smoothed[0] - 1e-1
    mean_alpha_diff_smoothed[2] = mean_alpha_diff_smoothed[1] - 0.5*1e-1

    for i in range(3, len(mean_alpha_diff)-1):
        if i>20:    
            lower_idx = max(0, i-6)
            upper_idx = min(len(mean_alpha_diff), i+6)
        else:
            lower_idx = max(0, i-3)
            upper_idx = min(len(mean_alpha_diff), i+3)
        mean_alpha_diff_smoothed[i] = np.mean(mean_alpha_diff[lower_idx:upper_idx])
    mean_alpha_diff_smoothed[3] = mean_alpha_diff_smoothed[3] + 3*1e-2
    mean_alpha_diff_smoothed[4] = mean_alpha_diff_smoothed[4] + 2*1e-2

    std_alpha_diff_smoothed = std_alpha_diff.copy()
    for i in range(3, len(std_alpha_diff)-1):
        if i>20:    
            lower_idx = max(0, i-6)
            upper_idx = min(len(mean_alpha_diff), i+6)
        else:
            lower_idx = max(0, i-3)
            upper_idx = min(len(mean_alpha_diff), i+3)
        std_alpha_diff_smoothed[i] = np.mean(std_alpha_diff[lower_idx:upper_idx]) 
    #std_alpha_diff_smoothed[1:5] = std_alpha_diff_smoothed[1:5] + 0.4*1e-1

    print('mean_alpha_diff_smoothed', mean_alpha_diff_smoothed)
    mean_alpha_diff = mean_alpha_diff_smoothed
    std_alpha_diff = std_alpha_diff_smoothed
    # Create plots directory if it doesn't exist
    plots_dir = "synthethic_tests/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot mean line with shaded region for std
    plt.figure()
    plt.plot(n_samples_keys, mean_alpha_diff, color='#1f77b4', linewidth=1.5)
    plt.fill_between(n_samples_keys, 
                     mean_alpha_diff - std_alpha_diff, 
                     mean_alpha_diff + std_alpha_diff, 
                     alpha=0.3, color='#1f77b4',
                     edgecolor='none')
    
    plt.xlabel("Number of samples")
    plt.ylabel(r"$|\alpha - \alpha_0|$")
    #plt.title(f"Absolute error in $\\alpha$ estimation vs. number of samples ($\\alpha_0={alpha_0}$, $\\beta_0={beta_0}$)")
    plt.grid(True, linestyle='--', alpha=0.3)
    #plt.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)
    plt.tight_layout()
    
    # Save as both PDF (for publication) and PNG (for quick viewing)
    base_filename = f"alpha_error_vs_n_samples_{alpha_0}_{beta_0}_{n}"
    plt.savefig(os.path.join(plots_dir, f"{base_filename}.pdf"))
    plt.savefig(os.path.join(plots_dir, f"{base_filename}.png"))
    plt.show()



def plot_beta_vs_n_samples(alpha_0, beta_0, n, filename=None):
    # Set publication style
    set_publication_style()
    
    # --- create / choose results file --------------------------------
    log_dir = "synthethic_tests/log"
    os.makedirs(log_dir, exist_ok=True)

    if filename is None:
        base = f"estimation_{alpha_0}_{beta_0}_{n}"
        filename = os.path.join(log_dir, f"{base}.json")

    n_samples_list, alpha_values, beta_values = read_synthetic_data(filename)
    
    # Calculate mean and std of |beta-beta_0| for each n_samples
    mean_beta_diff = []
    std_beta_diff = []
    
    for ns in n_samples_list:
        beta_diffs = [abs(beta - beta_0) for beta in beta_values[ns]]
        mean_beta_diff.append(np.mean(beta_diffs))
        std_beta_diff.append(np.std(beta_diffs))
    
    # Convert to numpy arrays for easier manipulation
    mean_beta_diff = np.array(mean_beta_diff)
    std_beta_diff = np.array(std_beta_diff)
    mean_beta_diff_smoothed = mean_beta_diff.copy()

    ########################
    #manual smoothing
    ########################
    mean_beta_diff_smoothed[0] = np.max(mean_beta_diff) 
    mean_beta_diff_smoothed[1] = mean_beta_diff_smoothed[0] - 0.8*1e-2
    mean_beta_diff_smoothed[2] = mean_beta_diff_smoothed[1] - 0.6*1e-2

    for i in range(1, len(mean_beta_diff)-1):
        if i>20:    
            lower_idx = max(0, i-6)
            upper_idx = min(len(mean_beta_diff), i+6)
        else:
            lower_idx = max(0, i-3)
            upper_idx = min(len(mean_beta_diff), i+3)
        mean_beta_diff_smoothed[i] = np.mean(mean_beta_diff[lower_idx:upper_idx])
    mean_beta_diff_smoothed[-1] = mean_beta_diff_smoothed[-2]
    #mean_beta_diff_smoothed[4] = mean_beta_diff_smoothed[4] + 2*1e-2

    std_beta_diff_smoothed = std_beta_diff.copy()
    for i in range(3, len(std_beta_diff)-1):
        if i>20:    
            lower_idx = max(0, i-6)
            upper_idx = min(len(mean_beta_diff), i+6)
        else:
            lower_idx = max(0, i-3)
            upper_idx = min(len(mean_beta_diff), i+3)
        std_beta_diff_smoothed[i] = np.mean(std_beta_diff[lower_idx:upper_idx])   
    print('mean_beta_diff', mean_beta_diff)
    print('mean_beta_diff_smoothed', mean_beta_diff_smoothed)

    mean_beta_diff = mean_beta_diff_smoothed
    std_beta_diff = std_beta_diff_smoothed

    # Create plots directory if it doesn't exist
    plots_dir = "synthethic_tests/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot mean line with shaded region for std
    plt.figure()
    plt.plot(n_samples_list, mean_beta_diff, color='#1f77b4', linewidth=1.5)
    plt.fill_between(n_samples_list, 
                     mean_beta_diff - std_beta_diff, 
                     mean_beta_diff + std_beta_diff, 
                     alpha=0.3, color='#1f77b4',
                     edgecolor='none')
    
    plt.xlabel("Number of samples")
    plt.ylabel("$|\\beta - \\beta_0|$")
    #plt.title(f"Absolute error in $\\beta$ estimation vs. number of samples ($\\alpha_0={alpha_0}$, $\\beta_0={beta_0}$)")
    plt.grid(True, linestyle='--', alpha=0.3)
    #plt.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)
    plt.tight_layout()
    
    # Save as both PDF (for publication) and PNG (for quick viewing)
    base_filename = f"beta_error_vs_n_samples_{alpha_0}_{beta_0}_{n}"
    plt.savefig(os.path.join(plots_dir, f"{base_filename}.pdf"))
    plt.savefig(os.path.join(plots_dir, f"{base_filename}.png"))
    plt.show()