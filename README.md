# Project Setup and Usage Guide

## 1. Installing Requirements

Install all necessary dependencies using the requirements file:

```bash
conda create -n myenv python=3.8
conda activate myenv
pip install -r requirements.txt
```

## 2. Kaggle Authentication for College Sport Data

To access college sport datasets, you need to authenticate with Kaggle:

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com) if you don't have one
2. Go to your account settings (click on your profile picture → "Account")
3. Scroll down to the API section and click "Create New API Token"
4. This downloads a `kaggle.json` file
5. Place this file in the `~/.kaggle/` directory:
   ```bash
   mkdir -p ~/.kaggle/
   cp path/to/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json  # Set appropriate permissions
   ```

## 3. Script Arguments

The script supports two modes: `fit` and `evaluate`, each with its own set of parameters:

### Available Arguments

| Mode | Parameter | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `fit` | `--dataset` | string | sushi | Dataset name to use |
| `fit` | `--n-teams` | integer | 100 | Number of teams to use |
| `fit` | `--delta` | float | 7 | Delta parameter value |
| `fit` | `--mc-samples` | integer | 100 | Number of Monte Carlo samples |
| `fit` | `--seed` | integer | 42 | Random seed for reproducibility |
| `fit` | `--n-trials` | integer | 1 | Number of trials to run |
| `fit` | `--save` | flag | False | Save results when enabled |
| `fit` | `--verbose` | flag | True | Enable verbose output |
| `evaluate` | `--dataset` | string | football | Dataset name for evaluation |
| `evaluate` | `--n-items` | integer | 100 | Number of items to evaluate |

## 4. Running the Script

### To fit models with default parameters:

```bash
python script.py fit
```

### To fit models with custom parameters:

```bash
python script.py fit --dataset sushi --n-teams 20 --delta 5.0 --mc-samples 200 --seed 123 --n-trials 3 --save
```

### To evaluate results with default parameters:

```bash
python script.py evaluate
```

### To evaluate with custom parameters:

```bash
python script.py evaluate --dataset football --n-items 50
```

### Getting help:

```bash
python script.py --help
```

For mode-specific help:

```bash
python script.py fit --help
```

or
```bash
python script.py evaluate --help
```

The output will be displayed in a fancy grid format for better readability, showing performance metrics across different models.