# Project Setup and Usage Guide
This repository contains the code for the paper *Mallows Model with Learned Distance Metrics: Sampling and Maximum Likelihood Estimation*.


## 0. Setup
First, clone the repository and navigate to the project directory:
```bash
git clone "link"
cd Generalized_Mallow_Learning
```

## 1. Installing Requirements

Install all necessary dependencies using the requirements file:

```bash
conda create -n myenv python
conda activate myenv
pip install -r requirements.txt
```

## 2. Kaggle Authentication for College Sport Data

To access college sport datasets, you need to have your Kaggle API credentials set.

## 3. Example Usage

### Real-World Dataset: sushi
```bash
python script.py fit-real-world --dataset sushi --n-trials 3 --verbose
```

### Real-World Dataset: football
```bash
python script.py fit-real-world --dataset football --n-trials 3 --verbose
```

### Real-World Dataset: basketball
```bash
python script.py fit-real-world --dataset basketball --n-trials 3 --verbose
```

### Synthetic Data:

```bash
python script.py fit-synthetic --n-items 15  --alpha-0 1.5 --beta-0 0.5 --n_train 10 50 200 --truncation 6 --n-trials 4 --verbose
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
| `--save` | flag | False | Save results when enabled |
| `--verbose` | flag | True | Enable verbose output |

### Synthetic Data Mode (`fit-synthetic`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n-items` | integer | 15 | Number of items (choices: 10, 15, 20) |
| `--alpha-0` | float | 1.5 | Alpha_0 parameter |
| `--beta-0` | float | 0.5 | Beta_0 parameter |
| `--n_train` | list of integers | [10, 50, 200] | Number of training samples |
| `--n-trials` | integer | 1 | Number of trials to run for each training size|
| `--truncation` | integer | 6 | Truncation parameter (choices: 3, 4, 5, 6) |
| `--save` | flag | False | Save results when enabled |
| `--verbose` | flag | True | Enable verbose output |

