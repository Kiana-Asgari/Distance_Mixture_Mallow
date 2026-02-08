# Maximum Likelihood Estimation (MLE) for Generalized Mallows Model
1. [Consensus Ranking Estimation](#1-consensus-ranking-estimation)
2. [Alpha-Beta Parameter Estimation](#2-alpha-beta-parameter-estimation)
3. [Score Function](#3-score-function)
4. [Top-k Evaluation Metrics](#4-top-k-evaluation-metrics)

## 1. Consensus Ranking Estimation

Given a set of $m$ observed rankings $\{\pi^{(1)}, \pi^{(2)}, \ldots, \pi^{(m)}\}$, the consensus ranking is

$$
\sigma^* = \arg\min_{\sigma \in S_n} \sum_{i=1}^{m} d_\alpha(\pi^{(i)}, \sigma).
$$

### Hungarian (Kuhn-Munkres) Algorithm Solution

The optimization can be reformulated as a linear sum assignment problem. We construct a cost matrix $C \in \mathbb{R}^{n \times n}$ where:

$$
C_{ij} = \sum_{k=1}^{m} |\pi^{(k)}_i - j|
\quad \Rightarrow\quad 
\sigma^* = \arg\min_{\sigma} \sum_{i=1}^{n} C_{i,\sigma(i)}
$$


## 2. Alpha-Beta Parameter Estimation

The log-likelihood for $m$ observations is:

$$
\ell(\alpha, \beta) = -m \log Z_n(\alpha, \beta) - \beta \sum_{i=1}^{m} d_\alpha(\pi^{(i)}, \sigma)
$$

Taking derivatives with respect to $\alpha$ and $\beta$:

$$
\frac{\partial \ell}{\partial \beta} = -m \frac{\partial \log Z_n}{\partial \beta} - \sum_{i=1}^{m} d_\alpha(\pi^{(i)}, \sigma)\; ,
\qquad
\frac{\partial \ell}{\partial \alpha} = -m \frac{\partial \log Z_n}{\partial \alpha} - \beta \sum_{i=1}^{m} \frac{\partial d_\alpha}{\partial \alpha}(\pi^{(i)}, \sigma)
$$

Using the identity $\frac{\partial \log Z}{\partial \beta} = -\mathbb{E}[d_\alpha]$ and rescaling by $\beta$,  the MLE equations become:
$$
\psi(\alpha, \beta) = \left( \mathbb{E}[d_\alpha(\pi, \sigma)] - \frac{1}{m}\sum_{i=1}^{m} d_\alpha(\pi^{(i)}, \sigma),\,
\mathbb{E}\left[\frac{\partial d_\alpha}{\partial \alpha}(\pi, \sigma)\right] - \frac{1}{m}\sum_{i=1}^{m} \frac{\partial d_\alpha}{\partial \alpha}(\pi^{(i)}, \sigma)\right)= 0.
$$

### Optimization Methods

#### 1. Least Squares (for $n \leq 20$)
$$
(\hat{\alpha}, \hat{\beta}) = \arg\min_{(\alpha, \beta)} \|\psi(\alpha, \beta)\|^2
$$


#### 2. Differential Evolution (for $n > 20$)

A global optimization algorithm that:
- Uses population-based search with mutation and recombination
- Employs Latin hypercube sampling for initial population
- Applies local polish refinement
- Minimizes: $\|\psi(\alpha, \beta)\|_{1/2}$


## 3. Score Function
### Numerical Computation of Expectations

Using finite differences with step size $h$:

$$
\mathbb{E}[d_\alpha] \approx -\frac{\log Z_n(\alpha, \beta + h) - \log Z_n(\alpha, \beta - h)}{2h}
$$

$$
\mathbb{E}\left[\frac{\partial d_\alpha}{\partial \alpha}\right] \approx -\frac{1}{\beta} \cdot \frac{\log Z_n(\alpha + h, \beta) - \log Z_n(\alpha - h, \beta)}{2h}
$$

### Lookup Table Interpolation

For computational efficiency with $n\geq 20$, precomputed lookup tables store $\mathbb{E}[d_\alpha]$ and $\mathbb{E}[\partial d_\alpha / \partial \alpha]$ on a grid of $(\alpha, \beta)$ values. During optimization, bivariate spline interpolation is used:

$$
\mathbb{E}[d_\alpha](\alpha, \beta) \approx S_1(\alpha, \beta)\;,
\quad
\mathbb{E}\left[\frac{\partial d_\alpha}{\partial \alpha}\right](\alpha, \beta) \approx S_2(\alpha, \beta)
$$


## 4. Evaluation Metrics

This module implements comprehensive ranking quality metrics for evaluating predicted rankings against ground truth.

### Input Format

- **Ground truth rankings**: $\{y^{(1)}, y^{(2)}, \ldots, y^{(m)}\}$
- **Predicted rankings**: $\{\hat{y}^{(1)}, \hat{y}^{(2)}, \ldots, \hat{y}^{(m)}\}$

Each ranking is a permutation of item indices.

### 4.1 Recall@k

Measures the proportion of relevant items (top-$k$ in ground truth) that appear in the top-$k$ of the prediction:

$$
\text{Recall@k} = \frac{|y_{\leq k} \cap \hat{y}_{\leq k}|}{k}
$$

where $y_{\leq k}$ denotes the set of top-$k$ items in $y$.

### 4.2 Precision@k

For ranking tasks, precision@k equals recall@k:

$$
\text{Precision@k} = \frac{|y_{\leq k} \cap \hat{y}_{\leq k}|}{k}
$$

### 4.3 Mean Reciprocal Rank (MRR)

Average of reciprocal ranks of the first relevant item:

$$
\text{MRR} = \frac{1}{m} \sum_{i=1}^{m} \frac{1}{\text{rank}_i}
$$

where $\text{rank}_i$ is the position of the first item from top-10 ground truth that appears in the predicted ranking:

$$
\text{rank}_i = \min\{j : \hat{y}^{(i)}_j \in y^{(i)}_{\leq 10}\}
$$

### 4.4 Normalized Discounted Cumulative Gain (NDCG@k)

Measures ranking quality with position-dependent relevance:

$$
\text{DCG@k} = \sum_{j=1}^{k} \frac{\text{rel}(\hat{y}_j)}{\log_2(j + 1)}
$$

where relevance is defined as:

$$
\text{rel}(item) = n - \text{position}_{true}(item)
$$

Normalized by the ideal DCG:

$$
\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}
$$

$$
\text{IDCG@k} = \sum_{j=1}^{k} \frac{n - j + 1}{\log_2(j + 1)}
$$

### 4.5 Hamming Distance

Normalized count of positions where rankings differ:

$$
\text{Hamming}(y, \hat{y}) = \frac{1}{n} \sum_{j=1}^{n} \mathbb{1}[y_j \neq \hat{y}_j]
$$

where $\mathbb{1}[\cdot]$ is the indicator function.

### 4.6 Pairwise Accuracy

Fraction of item pairs with correct relative order:

$$
\text{PairwiseAcc}(y, \hat{y}) = \frac{1}{\binom{n}{2}} \sum_{i<j} \mathbb{1}[\text{sgn}(y^{-1}(i) - y^{-1}(j)) = \text{sgn}(\hat{y}^{-1}(i) - \hat{y}^{-1}(j))]
$$

where $y^{-1}(i)$ gives the position of item $i$ in ranking $y$.

Equivalently:

$$
\text{PairwiseAcc} = \frac{\text{num. concordant pairs}}{\binom{n}{2}}
$$

### 4.7 Kendall's Tau

Correlation coefficient measuring rank correlation:

$$
\tau = \frac{n_c - n_d}{\binom{n}{2}}
$$

where:
- $n_c$ = number of concordant pairs
- $n_d$ = number of discordant pairs

A pair $(i, j)$ is concordant if both rankings agree on their relative order:

$$
\text{concordant: } (y^{-1}(i) - y^{-1}(j))(\hat{y}^{-1}(i) - \hat{y}^{-1}(j)) > 0
$$

Range: $\tau \in [-1, 1]$, where 1 indicates perfect agreement.

### 4.8 Spearman's Rho

Pearson correlation of rank positions:

$$
\rho = \frac{\text{cov}(y^{-1}, \hat{y}^{-1})}{\sigma_{y^{-1}} \sigma_{\hat{y}^{-1}}}
$$

Equivalent formula:

$$
\rho = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}
$$

where $d_i = y^{-1}(i) - \hat{y}^{-1}(i)$ is the difference in positions for item $i$.

Range: $\rho \in [-1, 1]$, where 1 indicates perfect monotonic relationship.
