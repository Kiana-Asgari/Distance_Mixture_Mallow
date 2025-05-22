# Project Overview and the Results
This repository contains the code for the paper *Mallows Model with Learned Distance Metrics: Sampling and Maximum Likelihood Estimation* ("arxive Link?").

In this work, we present a new statistical model for learning the underlying ranking (prefrence) from observed empirical rankings and comparing it to the other models. We also prove theoretical guarantees for convergence and the efficiency of our learning method. 

Models in this repository include:
• $L_\alpha$ Mallows (Distance Mixture model):
  $$P_{\alpha,\beta}(\pi) = \frac{1}{Z_{\alpha,\beta}} \exp\{ -\beta d_\alpha(\pi, \sigma_0) \} $$,
  where $d_\alpha(\pi, \sigma_0) = \sum_{i=1}^n |\pi(i) - \sigma_0(i)|^\alpha $
• Mallows's model with Kendall $\tau$ distance
• Plackett-Luce model

Our simulations show significant improvements achieved by learning the distance metric in the $L_\alpha$ Mallows model over famous models such as Plackett-Luce and Mallows's model with Kendall $\tau$ distance. The table below shows results for the college basketball dataset:

| Metric | $\boldsymbol{L}_{\boldsymbol{\alpha}}$-Mallows | Mallows's $\tau$ | Plackett-Luce |
|-------|-----------------|--------------|---------------|
| Estimated $\alpha$ | 0.002 ($\pm$ 0.002) | -- | -- |
| Estimated $\beta$ | 0.504 ($\pm$ 0.023) | -- | -- |
| $\uparrow$Spearman's $\rho$ correlation | **0.759** ($\pm$ 0.006) | 0.478 ($\pm$ 0.006) | 0.373 ($\pm$ 0.006) |
| $\uparrow$Kendall's $\tau$ correlation | **0.564** ($\pm$ 0.005) | 0.328 ($\pm$ 0.005) | 0.253 ($\pm$ 0.004) |
| $\downarrow$Hamming distance | **0.973** ($\pm$ 0.000) | 0.984 ($\pm$ 0.000) | 0.986 ($\pm$ 0.000) |
| $\uparrow$Pairwise accuracy (%) | **96.378** ($\pm$ 0.100) | 91.754 ($\pm$ 0.152) | 90.298 ($\pm$ 0.138) |
| $\uparrow$Top-1 hit rate (%) | **12.873** ($\pm$ 0.941) | 2.212 ($\pm$ 0.807) | 1.749 ($\pm$ 0.250) |
| $\uparrow$Top-5 hit rate (%) | **56.465** ($\pm$ 2.315) | 10.857 ($\pm$ 1.401) | 7.205 ($\pm$ 0.605) |

*Table: College basketball dataset for 100 teams, model out-of-sample performance averaged over 50 independent trials (mean ± standard deviation). Significant improvements are observed in the correlation metrics, pairwise accuracy, and the top hit rates.*

For the college football dataset, we observed similar improvements:

| Metric | $\boldsymbol{L}_{\boldsymbol{\alpha}}$-Mallows | Mallows's $\tau$ | Plackett-Luce |
|-------|-----------------|--------------|---------------|
| Estimated $\alpha$ | 0.003 ($\pm$ 0.002) | -- | -- |
| Estimated $\beta$ | 0.516 ($\pm$ 0.030) | -- | -- |
| $\uparrow$Spearman's $\rho$ correlation | **0.454** ($\pm$ 0.006) | 0.387 ($\pm$ 0.007) | 0.138 ($\pm$ 0.005) |
| $\uparrow$Kendall's $\tau$ correlation | **0.318** ($\pm$ 0.004) | 0.264 ($\pm$ 0.005) | 0.093 ($\pm$ 0.004) |
| $\downarrow$Hamming distance | **0.981** ($\pm$ 0.000) | 0.986 ($\pm$ 0.000) | 0.989 ($\pm$ 0.000) |
| $\uparrow$Pairwise accuracy (%) | **91.163** ($\pm$ 0.165) | 88.134 ($\pm$ 0.178) | 86.944 ($\pm$ 0.133) |
| $\uparrow$Top-1 hit rate (%) | **2.057** ($\pm$ 0.535) | 0.294 ($\pm$ 0.399) | 1.524 ($\pm$ 0.899) |
| $\uparrow$Top-5 hit rate (%) | **23.590** ($\pm$ 3.979) | 2.386 ($\pm$ 1.240) | 7.097 ($\pm$ 1.154) |

*Table: College football dataset for 100 teams, model out-of-sample performance averaged over 50 independent trials (mean ± standard deviation). Again, significant improvements are observed in the correlation metrics, pairwise accuracy, and the top hit rates*






# Usage Guide
## 0. Setup
First, clone the repository and navigate to the project directory:
```bash
git clone "link"
cd Generalized_Mallow_Learning
```

If you face difficulties with git, you can manually download and unzip the repository from the project page.

## 1. Installing Requirements

Install all necessary dependencies using the requirements file:

```bash
conda create -n myenv python
conda activate myenv
pip install -r requirements.txt
```

If you face issues with conda, you can alternatively use Python's built-in venv:

```bash
# on windows:
python -m venv myenv
myenv\Scripts\activate

# on mac/linux
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## 2. Kaggle Authentication for College Sport Data

To access college sport datasets, you need to have your Kaggle API credentials set.

## 3. Example Usage



### Real-World Dataset: basketball
```bash
python script.py fit-real-world --dataset basketball --n-teams 100 --n-trials 3 --verbose
```

```bash
python script.py fit-real-world --dataset basketball --n-teams 10 --n-trials 3 --verbose
```

### Real-World Dataset: football
```bash
python script.py fit-real-world --dataset football --n-trials 3 --verbose
```

### Real-World Dataset: sushi
```bash
python script.py fit-real-world --dataset sushi --n-trials 3 --verbose
```



### Synthetic Data:

```bash
python script.py fit-synthetic  --alpha-0 1.5 --beta-0 0.5 --n_train 50 --truncation_training 6 --n-trials 4 --verbose
```



## 4. Available Arguments
The script supports two modes:
- `fit-real-world`: Fits models on real-world datasets (sushi, football, basketball)
- `fit-synthetic`: Fits models on synthetic data

### Real-World Dataset Mode (`fit-real-world`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset` | string | sushi | Dataset name to use (sushi, football, basketball) |
| `--n-teams` | integer | 100 | Number of teams to use |
| `--truncation` | integer | 7 | Truncation parameter value (choices: 5, 6, 7) |
| `--mc-samples` | integer | 100 | Number of Monte Carlo samples for testing evaluation metrics |
| `--seed` | integer | 42 | Random seed for reproducibility |
| `--n-trials` | integer | 1 | Number of trials to run |
| `--verbose` | flag | True | Enable verbose output |

### Synthetic Data Mode (`fit-synthetic`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n-items` | integer | 15 | Number of items (choices: 10, 15, 20) |
| `--alpha-0` | float | 1.5 | Alpha_0 parameter |
| `--beta-0` | float | 0.5 | Beta_0 parameter |
| `--n_train` | list of integers | [10, 50, 200] | Number of training samples |
| `--n-trials` | integer | 1 | Number of trials to run for each training size|
| `--truncation_training` | integer | 6 | Truncation parameter for model training (choices: 3, 4, 5, 6) |
| `--truncation_data_generation` | integer | 8 | Truncation parameter for synthetic data generation |
| `--verbose` | flag | True | Enable verbose output |

