import numpy as np
from scipy.optimize import minimize


def learn_PL(permutations_train, permutations_test):
    # Fit Plackett-Luce model
    model = PlackettLuceModel(permutations_train)
    estimated_utilities = model.fit()
    
    # Compute normalized negative log-likelihood
    normalized_nll = model.compute_normalized_nll(permutations_test, estimated_utilities)
    return estimated_utilities, normalized_nll


class PlackettLuceModel:
    def __init__(self, rankings):
        """
        Initialize Plackett-Luce model with full ranking data
        
        Parameters:
        -----------
        rankings : numpy.ndarray
            2D array where each row represents a full ranking of items
            Shape: (n_rankings, n_items)
        """
        self.rankings = rankings
        self.n_items = rankings.shape[1]
    
    def negative_log_likelihood(self, params):
        """
        Compute negative log-likelihood for Plackett-Luce model
        
        Parameters:
        -----------
        params : numpy.ndarray
            Item utility parameters
        
        Returns:
        --------
        float: Negative log-likelihood
        """
        log_likelihood = 0
        
        for ranking in self.rankings:
            for i in range(len(ranking) - 1):  # -1 because last position is deterministic
                # Get utilities for remaining items
                remaining_items = ranking[i:]
                utilities = np.exp(params[remaining_items])
                
                # Probability of selecting item i from remaining items
                # P(πᵢ) = exp(θᵢ) / Σ(exp(θⱼ)) for remaining items j
                ranking_ll = np.log(utilities[0] / np.sum(utilities))
                log_likelihood += ranking_ll
        
        return -log_likelihood
    
    def fit(self, initial_guess=None):
        """
        Fit Plackett-Luce model using maximum likelihood estimation
        
        Parameters:
        -----------
        initial_guess : numpy.ndarray, optional
            Initial guess for item utilities. 
            If None, uses zeros as initial guess.
        
        Returns:
        --------
        numpy.ndarray: Estimated item utility parameters
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.n_items)
        
        # Bounds to ensure positive utilities
        bounds = [(None, None) for _ in range(self.n_items)]
        
        # Minimize negative log-likelihood
        result = minimize(
            self.negative_log_likelihood, 
            initial_guess, 
            method='L-BFGS-B',
            bounds=bounds
        )
        
        return result.x
    
    def predict_ranking_probability(self, ranking, utilities):
        """
        Compute probability of a specific ranking
        
        Parameters:
        -----------
        ranking : numpy.ndarray
            Specific ranking to compute probability for
        utilities : numpy.ndarray
            Estimated item utilities
        
        Returns:
        --------
        float: Probability of the given ranking
        """
        exp_utilities = np.exp(utilities[ranking])
        numerator = exp_utilities[0]
        denominator = np.sum(exp_utilities)
        return numerator / denominator
    
    def compute_normalized_nll(self, permutations_test, estimated_utilities):
        """
        Compute normalized negative log-likelihood for test permutations
        
        Parameters:
        -----------
        permutations_test : numpy.ndarray
            Test permutations to evaluate
            Shape: (n_test_permutations, n_items)
        estimated_utilities : numpy.ndarray
            Estimated item utility parameters from training
        
        Returns:
        --------
        float: Normalized negative log-likelihood
        """
        total_nll = 0
        
        for ranking in permutations_test:
            ranking_nll = 0
            for i in range(len(ranking) - 1):
                # Get utilities for remaining items
                remaining_items = ranking[i:]
                exp_utilities = np.exp(estimated_utilities[remaining_items])
                
                # Compute negative log-likelihood for each position
                nll = -np.log(exp_utilities[0] / np.sum(exp_utilities))
                ranking_nll += nll
            total_nll += ranking_nll
        
        # Normalize by number of test permutations
        return total_nll / len(permutations_test)
