# Why the held-out log-likelihood comparison is reported at n=10

The held-out log-likelihood in the response is reported for the n = 10
datasets (football, basketball, baseball, sushi, movie_lens, news) and
not for n = 100. Two reasons drive this choice.

First, the L_alpha Mallows partition function is approximated by a
banded-permanent DP whose truncation bandwidth needs to scale as
Delta = O(log(n / epsilon)) for a target TV error epsilon (Lemma 4.3).
At n = 10 the cap Delta = n - 1 = 9 is always reached and the
approximation is numerically exact; at n = 100 the recommended bandwidth
is closer to 14-20, which is an order of magnitude more expensive in
both time and memory. Empirically, going from Delta = 3 to Delta = 5 at
n = 100 with alpha = 1 and beta = 0.138 already changes log Z by more
than 25 log-units, so the truncation error dominates the reported
likelihood at that scale.

Second, at n = 100 the fitted alpha on real data sits at the
optimiser's lower bound (alpha hat = 0.1 +/- 0, documented separately in
the alpha-bound sensitivity experiment). At that alpha the L_alpha
distribution is essentially uniform because |x|^0.1 is close to 1 for
any nonzero x, so the log-likelihood is dominated by -log(n!) and
carries little discriminative information between methods.

We therefore evaluate log-likelihood at n = 10, where the banded-DP
approximation is tight and the learned alpha is most informative, and
retain Kendall tau and Top-1 hit rate at n = 100, matching the main
paper. This is a choice of evaluation methodology rather than a
limitation of the L_alpha Mallows model itself.
