"""
Plackett-Luce and Bradley-Terry model implementations.

Mathematical formulation:
- Plackett-Luce: P(π|θ) = ∏_{m=1}^{n-1} exp(θ_π[m]) / Σ_{j≥m} exp(θ_π[j])
- Bradley-Terry: Extracts all pairwise comparisons from rankings
"""
import numpy as np
from scipy.optimize import minimize


def learn_PL(train_rankings, test_rankings, lambda_reg: float = 0, BL_model: bool = False):
    """
    Fit Plackett-Luce or Bradley-Terry model using MLE.
    
    Args:
        train_rankings: Training rankings (n_rankings, n_items) with 0-based indices
        test_rankings: Test rankings for validation
        lambda_reg: Ridge regularization parameter (default: 0)
        BL_model: If True, use Bradley-Terry (all pairwise comparisons)
        
    Returns:
        utilities: Estimated item utilities (higher = better)
        test_nll: Normalized negative log-likelihood on test set
    """
    model = PlackettLuceModel(train_rankings, BL_model=BL_model)
    utilities = model.fit(lambda_reg=lambda_reg)
    test_nll = model.compute_normalized_nll(test_rankings, utilities)
    return utilities, test_nll


class PlackettLuceModel:
    """Plackett-Luce and Bradley-Terry model with MLE estimation."""
    
    def __init__(self, rankings: np.ndarray, BL_model: bool = False):
        """
        Args:
            rankings: (n_rankings, n_items) array where each row is a ranking
                     from best (column 0) to worst (last column)
                     Item indices are 0-based: 0, 1, 2, ..., n_items-1
            BL_model: If True, use Bradley-Terry (all pairwise comparisons)
        """
        self.rankings = rankings
        self.n_items = rankings.shape[1]
        self.BL_model = BL_model
        
        # Cache pair indices for Bradley-Terry (computed once)
        if BL_model:
            self._pair_indices = np.triu_indices(self.n_items, k=1)
            print("Bradley-Terry model is used\n")
 


    def negative_log_likelihood(self, params: np.ndarray, lambda_reg: float = 0.01) -> float:
        """
        Negative log-likelihood for Plackett-Luce or Bradley-Terry model.
        
        Plackett-Luce: NLL = -Σ_r Σ_m [θ_π[m] - log(Σ_{j≥m} exp(θ_π[j]))]
        Bradley-Terry: NLL = -Σ_r Σ_{m<k} log[exp(θ_π[m]) / (exp(θ_π[m]) + exp(θ_π[k]))]
        
        Both include ridge regularization: + λ/2 * ||θ||²
        """
        exp_theta = np.exp(params)
        exp_utils = exp_theta[self.rankings]  # (n_rankings, n_items)
        
        if self.BL_model:
            # Bradley-Terry: Vectorized extraction of all pairwise comparisons
            m_idx, k_idx = self._pair_indices
            
            # Get items at positions m and k for all rankings
            items_m = self.rankings[:, m_idx]  # (n_rankings, n_pairs)
            items_k = self.rankings[:, k_idx]  # (n_rankings, n_pairs)
            
            # Get utilities for these items
            theta_m = params[items_m]
            exp_m = exp_theta[items_m]
            exp_k = exp_theta[items_k]
            
            # Compute NLL for all pairs: -log P(m beats k)
            nll = np.sum(np.log(exp_m + exp_k) - theta_m)
        else:
            # Full Plackett-Luce model (already vectorized)
            denom = np.flip(np.cumsum(np.flip(exp_utils, axis=1), axis=1), axis=1)
            numer = exp_utils[:, :-1]
            denom = denom[:, :-1]
            nll_matrix = np.log(denom) - np.log(numer)
            nll = nll_matrix.sum()
        
        # Ridge regularization
        ridge_penalty = 0.5 * lambda_reg * np.sum(params**2)
        
        return nll + ridge_penalty

    def fit(self, lambda_reg: float = 0.01) -> np.ndarray:
        """
        Fit model using MLE with L-BFGS-B optimization.
        
        Args:
            lambda_reg: Ridge regularization parameter
            
        Returns:
            utilities: Estimated item utilities
        """
        initial_utilities = np.zeros(self.n_items)
        
        result = minimize(
            fun=lambda params: self.negative_log_likelihood(params, lambda_reg),
            x0=initial_utilities,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        return result.x

    @staticmethod
    def _matrix_nll(perms: np.ndarray, utilities: np.ndarray, BL_model: bool = False) -> float:
        """Compute negative log-likelihood for test set."""
        exp_theta = np.exp(utilities)
        exp_utils = exp_theta[perms]
        
        if BL_model:
            # Bradley-Terry: Vectorized pairwise comparison extraction
            n_rankings, n_items = perms.shape
            
            # Get all pairs (m, k) where m < k
            m_idx, k_idx = np.triu_indices(n_items, k=1)
            
            # Get items and their utilities
            items_m = perms[:, m_idx]
            items_k = perms[:, k_idx]
            theta_m = utilities[items_m]
            exp_m = exp_theta[items_m]
            exp_k = exp_theta[items_k]
            
            # Compute NLL
            nll = np.sum(np.log(exp_m + exp_k) - theta_m)
        else:
            # Full Plackett-Luce model
            denom = np.flip(np.cumsum(np.flip(exp_utils, axis=1), axis=1), axis=1)
            nll_matrix = np.log(denom[:, :-1]) - np.log(exp_utils[:, :-1])
            nll = nll_matrix.sum()
        
        return nll

    def compute_normalized_nll(self, test_rankings: np.ndarray, utilities: np.ndarray) -> float:
        """Compute normalized negative log-likelihood on test set."""
        total_nll = self._matrix_nll(test_rankings, utilities, self.BL_model)
        return total_nll / len(test_rankings)


def sample_PL(utilities: np.ndarray, n_samples: int = 1000, rng=None) -> np.ndarray:
    """
    Sample rankings from Plackett-Luce model using Gumbel-Max trick.
    
    Args:
        utilities: Item utilities (higher = more preferred)
        n_samples: Number of rankings to sample
        rng: Random number generator
        
    Returns:
        Array of shape (n_samples, n_items) with rankings from best to worst.
        Each ranking uses 0-based indices.
    """
    rng = rng or np.random.default_rng()
    
    # Add Gumbel noise to utilities
    gumbels = rng.gumbel(size=(n_samples, len(utilities)))
    scores = utilities + gumbels
    
    # Sort by descending score (best items first)
    return np.argsort(-scores, axis=1).astype(np.int64)
