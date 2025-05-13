import os
import numpy as np
import matplotlib.pyplot as plt

from synthethic_tests.synthethic_script import read_synthetic_data

def plot_alpha_vs_n_samples(alpha_0, beta_0, n, filename=None):
    # --- create / choose results file --------------------------------
    log_dir = "synthethic_tests/log"
    os.makedirs(log_dir, exist_ok=True)

    if filename is None:
        base = f"estimation_{alpha_0}_{beta_0}_{n}"
        filename = os.path.join(log_dir, f"{base}.json")

    n_samples_list, alpha_values, beta_values = read_synthetic_data(filename)
    
    # Calculate mean and std of |alpha-alpha_0| for each n_samples
    mean_alpha_diff = []
    std_alpha_diff = []
    
    for ns in n_samples_list:
        alpha_diffs = [abs(alpha - alpha_0) for alpha in alpha_values[ns]]
        mean_alpha_diff.append(np.mean(alpha_diffs))
        std_alpha_diff.append(np.std(alpha_diffs))
    
    # Convert to numpy arrays for easier manipulation
    mean_alpha_diff = np.array(mean_alpha_diff)
    std_alpha_diff = np.array(std_alpha_diff)
    
    # Create plots directory if it doesn't exist
    plots_dir = "synthethic_tests/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot mean line with shaded region for std
    plt.figure(figsize=(10, 6))
    plt.plot(n_samples_list, mean_alpha_diff, label=f"Mean |α-{alpha_0}|", color='blue')
    plt.fill_between(n_samples_list, 
                     mean_alpha_diff - std_alpha_diff, 
                     mean_alpha_diff + std_alpha_diff, 
                     alpha=0.3, color='blue',
                     label='±1 std dev')
    
    plt.xlabel("Number of samples")
    plt.ylabel("|α - α₀|")
    plt.title(f"Absolute error in α estimation vs number of samples (α₀={alpha_0}, β₀={beta_0})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"alpha_error_vs_n_samples_{alpha_0}_{beta_0}_{n}.png"))
    plt.show()



def plot_beta_vs_n_samples(alpha_0, beta_0, n, filename=None):
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

    # Create plots directory if it doesn't exist
    plots_dir = "synthethic_tests/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot mean line with shaded region for std
    plt.figure(figsize=(10, 6))
    plt.plot(n_samples_list, mean_beta_diff, label=f"Mean |β-{beta_0}|", color='blue')
    plt.fill_between(n_samples_list, 
                     mean_beta_diff - std_beta_diff, 
                     mean_beta_diff + std_beta_diff, 
                     alpha=0.3, color='blue',
                     label='±1 std dev')
    
    plt.xlabel("Number of samples")
    plt.ylabel("|β - β₀|")
    plt.title(f"Absolute error in β estimation vs number of samples (α₀={alpha_0}, β₀={beta_0})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"beta_error_vs_n_samples_{alpha_0}_{beta_0}_{n}.png"))
    plt.show()