# Pathway metric reconciliation

## Exact definitions

For subset `S`, let `W_S` contain the fixed unit-L2 PROGENy vectors and let

`P_S = W_S(W_S^T W_S)^+W_S^T`.

The biological-fidelity numerator is

`V_span(S) = sum_ci D6_ci^T P_S D14_ci`,

and `F(S)=V_span(S)/V_full` with `V_full=sum_ci D6_ci^T D14_ci`.

The recorded recoverability metric is calculated in the raw PROGENy score coordinates:

`T_raw(S)=sum_ci D6_ci^T W_S W_S^T D14_ci`,

`R_raw(S)=sum_ci (D6-Dhat6)_ci^T W_S W_S^T (D14-Dhat14)_ci`,

`g(S)=1-R_raw(S)/T_raw(S)`.

## Why `F(S)g(S)` is not reported

For `F(S)g(S)` to be the full-gene reproducible signal recovered inside the pathway span, `F` and `g` would have to use the same span numerator/denominator measure. They do not: `F` uses the orthogonal projector `W_S(W_S^TW_S)^+W_S^T`, whereas the fixed recorded `g` uses `W_SW_S^T`.

The pathway panel has maximum absolute off-diagonal Gram entry `0.931864832`. Thus the pathway vectors are materially non-orthogonal and the two quadratic forms are not interchangeable. Replacing `g` by whitened orthogonal coordinates defines a different prediction metric.

Singleton subsets are a special case because each fixed vector has unit norm, but that identity does not extend across the multi-pathway resolution curve.

**Decision:** `H=F*g` is algebraically invalid as a cross-resolution full-response recovery fraction. No `H` column or figure is produced, and no substitute composite score is invented.
