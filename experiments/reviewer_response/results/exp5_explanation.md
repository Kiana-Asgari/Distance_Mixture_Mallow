# Experiment 5 -- Table 2 high-precision audit
Re-extracted from the cached 50-trial JSON output of the main pipeline.
All numbers reported below are means over 50 trials, eight decimal places.

## baseball n=100
- our  : Kendall_tau = 0.56414497  Spearman_rho = 0.75751705  alpha = 0.10000015
- L1   : Kendall_tau = 0.56435997  Spearman_rho = 0.75770430  alpha = 1.00000000
- L2   : Kendall_tau = 0.56508186  Spearman_rho = 0.75830512  alpha = 2.00000000

## basketball n=100
- our  : Kendall_tau = 0.56414497  Spearman_rho = 0.75751705  alpha = 0.10000015
- L1   : Kendall_tau = 0.56435997  Spearman_rho = 0.75770430  alpha = 1.00000000
- L2   : Kendall_tau = 0.56508186  Spearman_rho = 0.75830512  alpha = 2.00000000

## Cross-file duplicate audit (Baseball n=100 vs Basketball n=100)
If `our`/`L1`/`L2`/`tau` numbers are identical across the two files, this indicates the basketball file inherited rows from the baseball run rather than a precision artefact of Table 2.

- our : baseball=0.56414497 basketball=0.56414497  --> **IDENTICAL**
- L1  : baseball=0.56435997 basketball=0.56435997  --> **IDENTICAL**
- L2  : baseball=0.56508186 basketball=0.56508186  --> **IDENTICAL**
- tau : baseball=0.23636725 basketball=0.23636725  --> **IDENTICAL**

## Recommendation

If the duplicate is genuine (alpha hat hits its lower bound, see Exp 3, and the consensus ranking is the same Borda center), a footnote in Table 2 should explain the collapse.  If the duplicate is a copy-paste artefact, regenerate the affected row with the n=100 Basketball pipeline and re-run.
