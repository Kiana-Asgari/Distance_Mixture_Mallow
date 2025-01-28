import numpy as np
import json
import os
from datasets.learn_american_football import get_full_rankings, get_top_teams
from datasets.load_american_football import load_data
from learning_params_new.learn_alpha import learn_beta_and_sigma
from learning_params_new.likelihood_test import test_error
from learning_params_new.learn_kendal import learn_kendal
from learning_params_new.learn_PL import learn_PL
import matplotlib.pyplot as plt


"""

logging the error, best beta, best sigma for each alpha
for each n_file and desired_teams there is a different json file


"""
def log_american_football_vs_alpha(n_file, n_top_teams, n_bottom_teams, 
                                   Delta, seed=42):
    # Create filename based on parameters
    filename = f'log_data/football_results_n{n_file}_1:top{n_top_teams}_bottom{n_bottom_teams}:-5.json'
    
    # Load existing data if file exists
    existing_data = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    
    np.random.seed(seed)  # For reproducibility

    teams, votes_dict = load_data(limit=n_file)

    top_teams = get_top_teams(teams, votes_dict)
    desired_teams = 1+np.concatenate([top_teams[1:n_top_teams], top_teams[-n_bottom_teams:-5]])
    full_rankings = get_full_rankings(teams, votes_dict, which_team_to_keep = desired_teams)
    print(f'full_rankings: {full_rankings.shape}')
    
    # Create 5 folds
    n_samples = full_rankings.shape[0]
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // 5

    alpha_list = np.linspace(1, 2, 100)

    for alpha in alpha_list:
        # Convert alpha to string for JSON key (with limited precision)
        alpha_key = f"{alpha:.6f}"
        
        # Skip if this alpha has already been processed
        if alpha_key in existing_data:
            print(f'Skipping alpha={alpha_key} (already processed)')
            #continue
            
        print(f'Processing alpha={alpha_key}...')
        
        # Initialize arrays to store results from each fold
        errors = []
        betas = []
        sigmas = []
        
        # Perform 5-fold cross validation
        for fold in range(5):
            #if fold!=0:
            #    continue
            # Create test indices for this fold
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < 4 else n_samples
            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            # Split data into train and test
            full_rankings_test = full_rankings[test_indices]
            full_rankings_train = full_rankings[train_indices]
            
            print(f'Fold {fold+1}: train shape: {full_rankings_train.shape}, test shape: {full_rankings_test.shape}')
            
            pi_0_hat, theta_hat, kendal_error = learn_kendal(full_rankings_train, full_rankings_test)

            print(f'Kendal: error: {kendal_error:3f}')
            print("Kendal: Estimated consensus ranking (pi_0):", pi_0_hat)
            print("Kendal: Estimated dispersion parameter (theta):", theta_hat)
        
            #theta_PL, nll_test = learn_PL(permutations_train=full_rankings_train,
            #         permutations_test=full_rankings_test)
           # print(f'PL: theta: {theta_PL}')
           # print(f'PL: nll_test: {nll_test}')

            if alpha_key in existing_data:
               print(f'Skipping alpha={alpha_key} (already processed)')
               continue
          

            beta_opt, sigma_opt = learn_beta_and_sigma(permutation_samples=full_rankings_train,
                                                     alpha=alpha,
                                                     beta_init=1,
                                                     Delta=Delta)
            
            # Test model
            error = test_error(full_rankings_test, beta_hat=beta_opt, sigma_hat=sigma_opt, alpha_hat=alpha)
            
            errors.append(error)
            betas.append(beta_opt)
            sigmas.append(sigma_opt)
        
        # Store average results across folds
        existing_data[alpha_key] = {
            'beta': float(np.mean(betas)),
            'beta_std': float(np.std(betas)),
            'sigma': np.mean(sigmas, axis=0).tolist(),  # Average sigma across folds
            'error': float(np.mean(errors)),
            'error_std': float(np.std(errors)),
            # Store individual fold results
            'fold_errors': [float(e) for e in errors],
            'fold_betas': [float(b) for b in betas],
            'fold_sigmas': [s.tolist() for s in sigmas],  # Store each fold's sigma
            'fold_details': [{  # Store detailed info for each fold
                'fold': i+1,
                'error': float(errors[i]),
                'beta': float(betas[i]),
                'sigma': sigmas[i].tolist()
            } for i in range(5)]
        }
        
        # Save after each iteration to prevent data loss
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)
            
        print(f'*for alpha={alpha}:')
        print(f'  Mean error: {np.mean(errors):.3f} ± {np.std(errors):.3f}')
        print(f'  Mean beta: {np.mean(betas):.3f} ± {np.std(betas):.3f}')
   






def plot_football_results(n_file, n_top_teams, n_bottom_teams):
    """
    Reads the football results from the JSON file and creates a plot showing
    error vs alpha with standard deviation bands.
    """
    
    # Create figures directory if it doesn't exist
    figures_dir = 'log_data/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Read the results
    filename = f'log_data/football_results_n{n_file}_1:top{n_top_teams}_bottom{n_bottom_teams}:-5.json'
    with open(filename, 'r') as f:
        results = json.load(f)
    
    # Convert data to lists for plotting
    alphas = [float(alpha) for alpha in results.keys()]
    errors = [results[alpha]['error'] for alpha in results.keys()]
    error_stds = [results[alpha]['error_std'] for alpha in results.keys()]
    
    # Sort all lists by alpha to ensure proper plotting
    sorted_indices = np.argsort(alphas)
    alphas = np.array(alphas)[sorted_indices]
    errors = -1 * np.array(errors)[sorted_indices]
    error_stds = np.array(error_stds)[sorted_indices]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the main line
    plt.plot(alphas, errors, 'b-', label='Mean Error', linewidth=2)
    
    # Add standard deviation bands
    plt.fill_between(alphas, 
                    errors - error_stds, 
                    errors + error_stds, 
                    alpha=0.2, 
                    color='b', 
                    label='±1 std')
    
    # Customize the plot
    plt.xlabel('Alpha', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title(f'Football Preference Error vs Alpha\n(n={n_file}, top={n_top_teams}, bottom={n_bottom_teams})', 
              fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(figures_dir, 
                f'football_error_vs_alpha_with_std_n{n_file}_top{n_top_teams}_bottom{n_bottom_teams}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find and print the minimum error point
    min_error_idx = np.argmin(errors)
    print(f"Minimum error: {errors[min_error_idx]:.4f} ± {error_stds[min_error_idx]:.4f}")
    print(f"at alpha = {alphas[min_error_idx]:.4f}")
   






import seaborn as sns
import matplotlib as mpl

def plot_football_first_fold(n_file, n_top_teams, n_bottom_teams):
    """
    Reads the football results from the JSON file and creates a plot showing
    error vs alpha for the first fold only.
    """
    sns.set_style("whitegrid", {'axes.edgecolor': 'darkgray',
                               'axes.linewidth': 0.7}) 
    mpl.rcParams.update({
        'text.usetex': True,            # For LaTeX rendering
        'font.family': 'serif',         # Use serif font family
        'font.serif': ['Computer Modern Roman'],  # Specific serif font
        'mathtext.fontset': 'cm',       # Use Computer Modern math font
        'figure.dpi': 120,              
        'figure.figsize': (7, 5),       
        'axes.labelsize': 17,           
        'axes.titlesize': 17,           
        'xtick.labelsize': 15,          
        'ytick.labelsize': 15,          
        'legend.fontsize': 15,          
        'lines.linewidth': 2,
        'axes.linewidth': 1.2,
        'font.size': 16,                
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{bm}', # Added bm package
        'mathtext.default': 'regular',   # Use regular (serif) font for math
        'axes.formatter.use_mathtext': True,  # Use mathtext for axis formatting
    })
    if n_top_teams==21 and n_bottom_teams==15:
        which_fold = 0
        PL_error = -51.268
        KL_error = -54.62679
        n_teams = 30
    elif n_top_teams==21 and n_bottom_teams==10:
        which_fold = 3
        PL_error = -39.621
        KL_error = -42.97
        n_teams = 25
    elif n_top_teams==11 and n_bottom_teams==15:
        which_fold = 3
        PL_error = -26.2841
        KL_error = -30.188
        n_teams = 20
    elif n_top_teams==11 and n_bottom_teams==10:
        which_fold = 4
        PL_error = -16.729
        KL_error = -18.809
        n_teams = 15
    else:
        which_fold = 0
        PL_error = 0  # You may want to set appropriate values for this case
        KL_error = 0  # You may want to set appropriate values for this case
        n_teams = 10
    # Create figures directory if it doesn't exist
    figures_dir = 'log_data/figures/football'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Read the results
    filename = f'log_data/football_results_n{n_file}_1:top{n_top_teams}_bottom{n_bottom_teams}:-5.json'
    with open(filename, 'r') as f:
        results = json.load(f)
    
    # Convert data to lists for plotting, using only first fold
    alphas = [float(alpha) for alpha in results.keys()]
    errors = [results[alpha]['fold_details'][which_fold]['error'] for alpha in results.keys()]  # Get first fold's error
    
    # Sort all lists by alpha to ensure proper plotting
    sorted_indices = np.argsort(alphas)
    alphas = np.array(alphas)[sorted_indices]
    errors = -1 * np.array(errors)[sorted_indices]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the line
    plt.plot(alphas, errors, 'b-',  linewidth=2, label=r'$L_\alpha$ Mallows model')
    
    # Add horizontal lines for PL and KL errors
    #if PL_error != 0:
    #    plt.axhline(y=-PL_error, color='darkred', linestyle='--', label=r'PL model')
    #if KL_error != 0:
    #    plt.axhline(y=-KL_error, color='red', linestyle='--', label=r'Kendall $\tau$ Mallows model')
    
    # Customize the plot
    plt.xlabel(r'$\alpha$', fontsize=12)
    plt.ylabel(r'average negative log-likelihood', fontsize=12)
    plt.title(r'{} teams from the American Football dataset'.format(n_teams), 
              fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(figures_dir, 
                f'football_errorn{n_file}_top{n_top_teams}_bottom{n_bottom_teams}.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find and print the minimum error point
    min_error_idx = np.argmin(errors)
    print(f"Minimum error (first fold): {errors[min_error_idx]:.4f}")
    print(f"at alpha = {alphas[min_error_idx]:.4f}")
   

