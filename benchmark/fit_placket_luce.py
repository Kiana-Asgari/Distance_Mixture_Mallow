import numpy as np
from scipy.optimize import minimize
from typing import Optional, List, Tuple

from real_world_datasets.utils import check_zero_based_index

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# =============================================================================
# Smart Initialization
# =============================================================================

def get_smart_init(rankings: np.ndarray) -> np.ndarray:
    """Frequency-based initialization for faster convergence."""
    n_items = rankings.shape[1]
    # Weight by position: items ranked higher get higher scores
    position_weights = 1.0 / (np.arange(n_items) + 1)
    scores = np.zeros(n_items)
    for pos in range(n_items):
        np.add.at(scores, rankings[:, pos], position_weights[pos])
    scores /= len(rankings)
    scores[0] = 0  # Fix first item
    return scores[1:] - scores[1:].mean()


# =============================================================================
# Core NLL and Gradient Computation
# =============================================================================

def compute_nll(rankings: np.ndarray, params_free: np.ndarray, lambda_reg: float = 0.0, 
                BL_model: bool = False) -> float:

    # Reconstruct full parameter vector with θ₀ = 0
    params = np.concatenate([[0.0], params_free])
    
    exp_theta = np.exp(params)
    exp_utils = exp_theta[rankings]
    
    if BL_model:
        # Bradley-Terry: P(i > j) = exp(θ_i) / (exp(θ_i) + exp(θ_j))
        exp_utils_safe = np.maximum(exp_utils, 1e-300)
        nll = np.log(exp_utils_safe[:, 0] + exp_utils_safe[:, 1]) - np.log(exp_utils_safe[:, 0])
    else:
        # Plackett-Luce: sequential choice model - numerical stability
        denom = np.flip(np.cumsum(np.flip(exp_utils, axis=1), axis=1), axis=1)
        denom = np.maximum(denom, 1e-300)
        exp_utils_safe = np.maximum(exp_utils, 1e-300)
        nll = np.log(denom[:, :-1]) - np.log(exp_utils_safe[:, :-1])
    
    # Ridge regularization (fixed bug)
    ridge = 0.5 * lambda_reg * np.sum(params_free**2)
    return nll.sum() / len(rankings) + ridge


def compute_gradient(rankings: np.ndarray, params_free: np.ndarray, lambda_reg: float = 0.0,
                     BL_model: bool = False) -> np.ndarray:

    # Reconstruct full parameter vector with θ₀ = 0
    params = np.concatenate([[0.0], params_free])
    n_items = len(params)
    
    exp_theta = np.exp(params)
    exp_utils = exp_theta[rankings]
    grad_full = np.zeros(n_items)
    
    if BL_model:
        exp_utils_safe = np.maximum(exp_utils, 1e-300)
        first, second = rankings[:, 0], rankings[:, 1]
        probs = exp_utils_safe[:, 1] / (exp_utils_safe[:, 0] + exp_utils_safe[:, 1])
        np.add.at(grad_full, first, probs)
        np.add.at(grad_full, second, -probs)
    else:
        denom = np.flip(np.cumsum(np.flip(exp_utils, axis=1), axis=1), axis=1)
        denom = np.maximum(denom, 1e-300)  # Numerical stability
        probs = exp_utils / denom
        
        # Vectorized: subtract 1 for all chosen items at each position
        np.add.at(grad_full, rankings[:, :-1].ravel(), -1)
        
        # Vectorized: add probabilities for all remaining items
        for m in range(n_items - 1):
            np.add.at(grad_full, rankings[:, m:].ravel(), probs[:, m:].ravel())
    
    # Extract gradient for free parameters (exclude θ₀) and add regularization
    # CRITICAL: Average the gradient to match the averaged loss
    grad_free = grad_full[1:] / len(rankings) + lambda_reg * params_free
    
    return grad_free


# =============================================================================
# Combined NLL + Gradient (avoids redundant computation)
# =============================================================================

def compute_nll_and_grad(rankings: np.ndarray, params_free: np.ndarray, 
                         lambda_reg: float = 0.0, BL_model: bool = False) -> Tuple[float, np.ndarray]:
    """Compute both NLL and gradient in one pass to avoid redundant exponential calculations."""
    
    params = np.concatenate([[0.0], params_free])
    n_items = len(params)
    
    exp_theta = np.exp(params)
    exp_utils = exp_theta[rankings]
    grad_full = np.zeros(n_items)
    
    if BL_model:
        # NLL - numerical stability
        exp_utils_safe = np.maximum(exp_utils, 1e-300)
        nll = np.log(exp_utils_safe[:, 0] + exp_utils_safe[:, 1]) - np.log(exp_utils_safe[:, 0])
        
        # Gradient
        first, second = rankings[:, 0], rankings[:, 1]
        probs = exp_utils_safe[:, 1] / (exp_utils_safe[:, 0] + exp_utils_safe[:, 1])
        np.add.at(grad_full, first, probs)
        np.add.at(grad_full, second, -probs)
    else:
        # NLL - Numerical stability: clip to avoid log(0)
        denom = np.flip(np.cumsum(np.flip(exp_utils, axis=1), axis=1), axis=1)
        denom = np.maximum(denom, 1e-300)
        exp_utils_safe = np.maximum(exp_utils, 1e-300)
        nll = np.log(denom[:, :-1]) - np.log(exp_utils_safe[:, :-1])
        
        # Gradient (vectorized)
        probs = exp_utils / denom
        np.add.at(grad_full, rankings[:, :-1].ravel(), -1)
        for m in range(n_items - 1):
            np.add.at(grad_full, rankings[:, m:].ravel(), probs[:, m:].ravel())
    
    # Regularization (fixed bug: was missing params_free**2)
    ridge = 0.5 * lambda_reg * np.sum(params_free**2)
    total_nll = nll.sum() / len(rankings) + ridge
    # CRITICAL: Average the gradient to match the averaged loss
    grad_free = grad_full[1:] / len(rankings) + lambda_reg * params_free
    
    return total_nll, grad_free


# =============================================================================
# Model Fitting
# =============================================================================

def fit_PL(rankings: np.ndarray, lambda_reg: float = 0.0, BL_model: bool = False) -> np.ndarray:

    n_items = rankings.shape[1]
    
    # Smart initialization for faster convergence
    x0 = get_smart_init(rankings)
    
    # Cache to avoid redundant computation when optimizer calls fun then jac with same params
    cache = {'params': None, 'nll': None, 'grad': None}
    
    def objective(p):
        if cache['params'] is None or not np.array_equal(p, cache['params']):
            cache['nll'], cache['grad'] = compute_nll_and_grad(rankings, p, lambda_reg, BL_model)
            cache['params'] = p.copy()
        return cache['nll']
    
    def gradient(p):
        if cache['params'] is None or not np.array_equal(p, cache['params']):
            cache['nll'], cache['grad'] = compute_nll_and_grad(rankings, p, lambda_reg, BL_model)
            cache['params'] = p.copy()
        return cache['grad']
    
    result = minimize(
        fun=objective,
        x0=x0,
        jac=gradient,
        method='L-BFGS-B',
        options={
            'maxiter': 500,
            'maxcor': 20,      # More memory for better convergence
            'ftol': 1e-6,     # Tighter function tolerance
            'gtol': 1e-4,      # Tighter gradient tolerance
            'disp': False
        }
    )
    
    # Return full parameter vector with θ₀ = 0
    return np.concatenate([[0.0], result.x])


def evaluate_nll(rankings: np.ndarray, utilities: np.ndarray, BL_model: bool = False) -> float:
    """
    Evaluate normalized NLL on test data (no regularization).
    
    utilities should be the full parameter vector (including θ₀ = 0).
    """
    # Extract free parameters (exclude θ₀)
    params_free = utilities[1:]
    nll = compute_nll(rankings, params_free, lambda_reg=0.0, BL_model=BL_model)
    return nll / len(rankings)


# =============================================================================
# Cross-Validation
# =============================================================================

def cross_validate_lambda(rankings: np.ndarray, lambda_candidates: List[float],
                          n_folds: int = 5, BL_model: bool = False,
                          random_seed: Optional[int] = None, n_jobs: int = 1) -> float:
    """Select optimal lambda via k-fold cross-validation (parallel if joblib available)."""
    n_samples = len(rankings)
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(n_samples)
    
    # Create folds
    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[:n_samples % n_folds] += 1
    folds = np.split(indices, np.cumsum(fold_sizes)[:-1])
    
    def eval_fold(k, lam):
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
        val_idx = folds[k]
        utils = fit_PL(rankings[train_idx], lambda_reg=lam, BL_model=BL_model)
        return evaluate_nll(rankings[val_idx], utils, BL_model)
    
    best_lambda, best_score = None, float('inf')
    
    for lam in lambda_candidates:
        # Parallel if joblib available and n_jobs > 1
        if HAS_JOBLIB and n_jobs != 1:
            fold_scores = Parallel(n_jobs=n_jobs)(
                delayed(eval_fold)(k, lam) for k in range(n_folds)
            )
        else:
            fold_scores = [eval_fold(k, lam) for k in range(n_folds)]
        
        avg_score = np.mean(fold_scores)
        print(f"  λ={lam:.4f}: validation NLL = {avg_score:.6f} ± {np.std(fold_scores):.6f}")
        
        if avg_score < best_score:
            best_score, best_lambda = avg_score, lam
    
    return best_lambda


# =============================================================================
# Main API
# =============================================================================

def learn_PL(permutations_train: np.ndarray, permutations_test: np.ndarray,
             use_cv: bool = False, lambda_reg: float = 0.0,
             lambda_candidates: Optional[List[float]] = None,
             n_folds: int = 5, BL_model: bool = False, n_jobs: int = 1) -> Tuple[np.ndarray, float, Optional[float]]:

    train = check_zero_based_index(permutations_train)
    test = check_zero_based_index(permutations_test)
    
    if BL_model:
        print("Using Bradley-Terry model\n")
    
    optimal_lambda = None
    if use_cv:
        if lambda_candidates is None:
            lambda_candidates = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        optimal_lambda = cross_validate_lambda(train, lambda_candidates, n_folds, BL_model, n_jobs=n_jobs)
        print(f"\nOptimal λ = {optimal_lambda}\n")
        lambda_reg = optimal_lambda
    
    utilities = fit_PL(train, lambda_reg, BL_model)
    test_nll = evaluate_nll(test, utilities, BL_model)
    
    return utilities, test_nll, optimal_lambda


# =============================================================================
# Sampling Utilities
# =============================================================================

def sample_PL(utilities: np.ndarray, n_samples: int = 1000, rng=None) -> np.ndarray:
    """Sample rankings from Plackett-Luce model using Gumbel-Max trick."""
    rng = rng or np.random.default_rng()
    gumbels = rng.gumbel(size=(n_samples, len(utilities)))
    return np.argsort(-(utilities + gumbels), axis=1).astype(np.int64)
