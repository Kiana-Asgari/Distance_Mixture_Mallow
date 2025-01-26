import numpy as np
from datasets.load_american_football import load_data

from learning_params_new.learn_alpha import learn_beta_and_sigma
from learning_params_new.likelihood_test import test_error
import sys



# good one so far: 1+top_teams[1:21], 1+top_teams[-20:-5]
# 1+np.concatenate([top_teams[1:21], top_teams[-15:-5]]):0.19
# 1+np.concatenate([top_teams[1:21], top_teams[-10:-5]]):0.23
# 1+np.concatenate([top_teams[1:11], top_teams[-15:-5]]):0.34
def learn_american_football():
    print('********************learning american football********************')
    print(' for the first 20 teams')
    teams, votes_dict = load_data(limit=100)
    top_teams = get_top_teams(teams, votes_dict)
    desired_teams = 1+np.concatenate([top_teams[1:11], top_teams[-15:-5]])
    full_rankings = get_full_rankings(teams, votes_dict, which_team_to_keep = desired_teams) #keep 20 first teams
    print(f'full_rankings: {full_rankings.shape}')
    
    # Randomly select 50 rankings for the test set
    np.random.seed(42)  # For reproducibility
    test_indices = np.random.choice(full_rankings.shape[0], 50, replace=False)
    full_rankings_test = full_rankings[test_indices]
    
    # Use the remaining rankings for training
    train_indices = np.setdiff1d(np.arange(full_rankings.shape[0]), test_indices)
    full_rankings_train = full_rankings[train_indices]

    print(f'rankings train: {full_rankings_train.shape}')
    print(f'rankings test: {full_rankings_test.shape}')

    for alpha in np.linspace(1, 1.5, 10):
        beta_opt, sigma_opt = learn_beta_and_sigma(permutation_samples=full_rankings_train,
                                                    alpha=alpha,
                                                    beta_init=1,
                                                    Delta=11)
        error = test_error(full_rankings_test, beta_hat=beta_opt, sigma_hat=sigma_opt, alpha_hat=alpha)
        print(f'*for alpha={alpha}, beta_opt: {beta_opt:3f}, error: {error:3f}, sigma_opt: {sigma_opt}')
    sys.exit()


    p_hat, theta_hat, sigma_hat, result = fit_mallow(permutation_samples=full_rankings_train, 
                                                        p_init=np.exp(2),
                                                        theta_init=np.exp(-2.0),
                                                        sigma_init=np.arange(20), 
                                                        Delta=5,
                                                        max_iter=5000, 
                                                        tol=1e-6,
                                                        verbose=True,
                                                        seed=42)
    beta_hat = -1*np.log(theta_hat)
    alpha_hat = np.log(p_hat)
    print(f'alpha_hat: {alpha_hat}')
    print(f'beta_hat: {beta_hat}')
    print(f'sigma_hat: {sigma_hat}')    












def get_full_rankings(teams, votes_dict, which_team_to_keep=None):
    full_rankings = []
    n = len(teams)
    if which_team_to_keep is None:
        which_team_to_keep = 10+ np.arange(20)

    for key in votes_dict.keys():
        _votes = np.array(votes_dict[key])

        for vote in _votes:

            # Filter the vote to only include teams in which_team_to_keep
            filtered_vote = [team for team in vote if team in which_team_to_keep]
            # Create a mapping from original team indices to new indices
            team_to_new_index = {team: i for i, team in enumerate(which_team_to_keep)}
            
            # Adjust the filtered vote to be a ranking from 0 to len(which_team_to_keep) - 1
            adjusted_vote = [team_to_new_index[team] for team in filtered_vote]
            if len(np.unique(adjusted_vote)) == len(which_team_to_keep) and\
                                                len(adjusted_vote) == len(which_team_to_keep):
                full_rankings.append(adjusted_vote)


    return np.array(full_rankings)





def get_top_teams(teams, votes_dict, top_n=20):

    votes=[] 
    for key in votes_dict.keys():
        votes+=votes_dict[key]
    num_teams=len(teams)
    rank_probabilities = np.zeros((num_teams, num_teams))
    # --------------------------------------
    # 2) Process Votes
    # --------------------------------------
    for vote in votes:
        for rank, team in enumerate(vote):
            rank_probabilities[team - 1, rank] += 1  # Adjust for 0-based indexing
    #rank_probabilities/= np.array(votes).shape[0]

    rank_order = np.argmax(rank_probabilities, axis=0)

    # Remove duplicates while maintaining order
    _, unique_indices = np.unique(rank_order, return_index=True)
    unique_rank_order = rank_order[np.sort(unique_indices)]

    # Convert unique_rank_order to a list of indices
    unique_rank_order = list(unique_rank_order)

    #print('teams: ', teams)
    #print('unique_rank_order: ', unique_rank_order)
    #print('top_teams: ', [teams[i] for i in unique_rank_order])  # Use list comprehension to index

    return np.array(unique_rank_order)

