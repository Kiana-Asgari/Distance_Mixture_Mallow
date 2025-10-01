import numpy as np
from scipy.optimize import minimize
from typing import Optional


# ----------------------------------------------------------------------
#  High-level helper
# ----------------------------------------------------------------------
def learn_PL(permutations_train, permutations_test, lambda_reg: float = 0, BL_model: bool = False):
    """
    Fit a Plackett–Luce model and return (utilities , normalised-NLL on test set).
    If BL_model=True, reduces to Bradley-Terry model (pairwise comparisons only).
    
    Parameters
    ----------
    permutations_train : np.ndarray
        Training permutations
    permutations_test : np.ndarray
        Test permutations
    lambda_reg : float, default=0.01
        Ridge regularization parameter
    BL_model : bool, default=True
        If True, reduces to Bradley-Terry model (pairwise comparisons only)
    """
    model = PlackettLuceModel(permutations_train, BL_model=BL_model)
    est_utils = model.fit(lambda_reg=lambda_reg)
    nll = model.compute_normalized_nll(permutations_test, est_utils)
    return est_utils, nll


# ----------------------------------------------------------------------
#  Plackett–Luce model class
# ----------------------------------------------------------------------
class PlackettLuceModel:

    def __init__(self, rankings: np.ndarray, BL_model: bool = False):
        """
        Parameters
        ----------
        rankings : (n_rankings , n_items) int ndarray
            Each row is a full ranking from *best* (col 0) to *worst* (last col).
            Item values are indices 0 … n_items-1.
        BL_model : bool, default=False
            If True, reduces to Bradley-Terry model (pairwise comparisons only)
        """
        self.rankings = rankings
        self.n_items  = rankings.shape[1]
        self.BL_model = BL_model
        if BL_model:
            print("Bradley-Terry model is used\n\n")
 


    def negative_log_likelihood(self, params: np.ndarray, lambda_reg: float = 0.01) -> float:
        """
        Vectorised implementation of

            L(θ) = − Σ_r Σ_{m=0}^{n−2} log  exp(θ_{π_r[m]}) /
                                                Σ_{j≥m} exp(θ_{π_r[j]})
                    + λ/2 * ||θ||²

        where π_r is ranking r and λ is the ridge regularization parameter.
        
        If BL_model=True, reduces to Bradley-Terry model (pairwise comparisons only):
            L(θ) = − Σ_r log exp(θ_{π_r[0]}) / (exp(θ_{π_r[0]}) + exp(θ_{π_r[1]}))
        """
        exp_theta   = np.exp(params)                   # (n_items,)
        exp_utils   = exp_theta[self.rankings]         # (R , n)

        if self.BL_model:
            # Bradley-Terry model: only consider pairwise comparisons (first two items)
            # P(i > j) = exp(θ_i) / (exp(θ_i) + exp(θ_j))
            first_item_utils = exp_utils[:, 0]         # (R,)
            second_item_utils = exp_utils[:, 1]        # (R,)
            
            # For Bradley-Terry: log P(first > second) = log(exp(θ_first) / (exp(θ_first) + exp(θ_second)))
            # = θ_first - log(exp(θ_first) + exp(θ_second))
            nll_matrix = np.log(first_item_utils + second_item_utils) - np.log(first_item_utils)
        else:
            # Full Plackett-Luce model
            # Denominator: reverse cumulative sums so that
            # denom[r, m] = Σ_{j=m}^{n−1} exp(θ_{π_r[j]})
            denom = np.flip(np.cumsum(np.flip(exp_utils, axis=1), axis=1), axis=1)

            # Exclude the last deterministic choice (prob = 1)
            numer       = exp_utils[:, :-1]
            denom       = denom[:, :-1]

            nll_matrix  = np.log(denom) - np.log(numer)    # (R , n−1)
        
        # Add ridge regularization term
        ridge_term = 0.5 * lambda_reg * np.sum(params**2)
        
        return nll_matrix.sum() + ridge_term       # scalar

    # -----------------------------
    #  MLE fit
    # -----------------------------
    def fit(self, initial_guess: Optional[np.ndarray] = None, lambda_reg: float = 0.01) -> np.ndarray:
        if initial_guess is None:
            initial_guess = np.zeros(self.n_items)

        # Create a wrapper function that passes the regularization parameter
        def objective(params):
            return self.negative_log_likelihood(params, lambda_reg)

        res = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'disp': False}
        )
        return res.x

    # -----------------------------
    #  Vectorised test NLL
    # -----------------------------
    @staticmethod
    def _matrix_nll(perms: np.ndarray, utilities: np.ndarray, BL_model: bool = False) -> float:
        """Shared helper used by both train- and test-NLL routines."""
        exp_theta = np.exp(utilities)
        exp_utils = exp_theta[perms]
        
        if BL_model:
            # Bradley-Terry model: only consider pairwise comparisons (first two items)
            first_item_utils = exp_utils[:, 0]         # (R,)
            second_item_utils = exp_utils[:, 1]        # (R,)
            nll = np.log(first_item_utils + second_item_utils) - np.log(first_item_utils)
        else:
            # Full Plackett-Luce model
            denom = np.flip(np.cumsum(np.flip(exp_utils, axis=1), axis=1), axis=1)
            nll = np.log(denom[:, :-1]) - np.log(exp_utils[:, :-1])
        
        return nll.sum()

    def compute_normalized_nll(
        self, permutations_test: np.ndarray, estimated_utilities: np.ndarray
    ) -> float:
        total_nll = self._matrix_nll(permutations_test, estimated_utilities, self.BL_model)
        return total_nll / permutations_test.shape[0]


# ----------------------------------------------------------------------
#  Fully-vectorised Plackett–Luce sampler (Gumbel–Max)
# ----------------------------------------------------------------------
def sample_PL(utilities: np.ndarray, n_samples: int = 1000, rng=None) -> np.ndarray:
    """
    Draw `n_samples` permutations *simultaneously* using the Gumbel–Max trick.

    Each row is a ranking from best (col 0) to worst (last col).
    """
    rng = rng or np.random.default_rng()
    gumbels   = rng.gumbel(size=(n_samples, len(utilities)))
    scores    = utilities + gumbels                      # broadcast add
    return np.argsort(-scores, axis=1).astype(np.int64)  # descending ⇒ best first
