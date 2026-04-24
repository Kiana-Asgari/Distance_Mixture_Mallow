# GitHub Actions workflows

## `run_experiments.yml` -- Run the experiment suite

Manually-triggered workflow that runs `experiments/run_all.py` on a
GitHub-hosted Ubuntu runner. Every run, whether it succeeds or fails,
uploads the produced CSVs, PDFs, and full log as a build artifact.

### Required secrets

Set these in **Settings -> Secrets and variables -> Actions**:

| Secret           | Value                                                           |
| ---------------- | --------------------------------------------------------------- |
| `KAGGLE_USERNAME` | Your Kaggle username (the slug from your profile URL).          |
| `KAGGLE_KEY`      | 32-character hex key from `kaggle.json` (created at <https://www.kaggle.com/settings> -> API -> Create New Token). |

They are used to authenticate `kagglehub`, which pulls the Massey
Ratings dataset for the football / basketball / baseball experiments.
If either secret is missing the workflow still runs, with the sports
datasets skipped and a warning annotation emitted.

Credentials never appear in the workflow YAML; they are read from
`${{ secrets.* }}` only, and the step that writes `~/.kaggle/kaggle.json`
builds the file through a Python heredoc so the values are never echoed
to the shell log.

### Triggering a run

From the web UI: **Actions** tab -> **Run experiments** -> **Run
workflow**.

Inputs:

| Input      | Type    | Default | Meaning                                                                |
| ---------- | ------- | ------- | ---------------------------------------------------------------------- |
| `n_trials` | string  | `50`    | Trials per experiment. Ignored when `quick` is true.                   |
| `only`     | string  | *empty* | Comma-separated script names (see `experiments/README.md`).            |
| `quick`    | boolean | `false` | Run every selected script with a reduced grid for quick validation.    |

From the command line with the GitHub CLI:

```
gh workflow run run_experiments.yml \
    -f n_trials=50 \
    -f only= \
    -f quick=false
```

A small smoke run to validate the setup before committing to a long run:

```
gh workflow run run_experiments.yml -f quick=true
```

### Finding the results

1. Open the run under the **Actions** tab.
2. The run summary page shows:
   * the inputs you selected,
   * row counts per script (via `check_status.py`),
   * which output files exist,
   * any script that crashed.
3. Scroll to the bottom of the run page: under **Artifacts** download
   `experiment-outputs-<run id>.zip`. It contains
   * `experiments/results/` -- all CSVs and the aggregate-metrics notes,
   * `experiments/figures/` -- all PDFs,
   * `experiments/run.log` -- full stdout / stderr.

Artifacts are retained for 30 days; download or re-upload anything you
want to keep longer.

### Timeout

The job has a 350-minute timeout (GitHub-hosted runners allow up to 360).
A full `--n-trials 50` run across all 11 dataset specs may take several
hours; use `only` to stage shorter runs while iterating.
