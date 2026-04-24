# Aggregate-metrics audit

Re-extracted from the cached 50-trial JSON output in `real_world_datasets/results/`. All numbers below are means over 50 trials, reported at eight decimal places.

## baseball, n=100
- `our  `  Kendall tau = 0.56414497  Spearman rho = 0.75751705  alpha hat = 0.10000015
- `L1   `  Kendall tau = 0.56435997  Spearman rho = 0.75770430  alpha hat = 1.00000000
- `L2   `  Kendall tau = 0.56508186  Spearman rho = 0.75830512  alpha hat = 2.00000000

## basketball, n=100
- `our  `  Kendall tau = 0.56414497  Spearman rho = 0.75751705  alpha hat = 0.10000015
- `L1   `  Kendall tau = 0.56435997  Spearman rho = 0.75770430  alpha hat = 1.00000000
- `L2   `  Kendall tau = 0.56508186  Spearman rho = 0.75830512  alpha hat = 2.00000000

## Cross-dataset identity check (baseball n=100 vs basketball n=100)

Two independent datasets producing bit-identical means across 50 trials would be a data-integrity signal rather than a rounding artefact.

- `our `: baseball=0.56414497  basketball=0.56414497  -> **IDENTICAL**
- `L1  `: baseball=0.56435997  basketball=0.56435997  -> **IDENTICAL**
- `L2  `: baseball=0.56508186  basketball=0.56508186  -> **IDENTICAL**
- `tau `: baseball=0.23636725  basketball=0.23636725  -> **IDENTICAL**
