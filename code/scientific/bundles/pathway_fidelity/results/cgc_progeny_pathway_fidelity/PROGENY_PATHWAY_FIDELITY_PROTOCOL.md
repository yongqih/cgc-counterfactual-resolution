# PROGENy five-pathway oracle-fidelity protocol

Fixed before computing any new pathway-span fidelity outcome.

## Provenance and scope

- Fixed scientific provenance: `46b25eec4a97e3001cc43262980af8f69c533478`.
- Valid primary-budget result to be contextualized, not rerun: full-gene recovery `0.11068053088808294`; pooled five-pathway recovery `0.27281393788398667` at `m=49,k=92`.
- Exact pathways and order: `MAPK`, `PI3K`, `JAK-STAT`, `p53`, `NFkB`.
- Exact 25,695-gene loading source SHA-256: `f5c4faa94fe03b2d088b72743da175380dfcf2edbfacd0077cc25eb19d2ca7e9`.
- Exact gene metadata SHA-256: `f690be9e4a57bcf1fc2c70d1e458ea901950c5c42cc84c6d1fb7e728094c4960`.
- Exact primary-gene indices SHA-256: `efea7718149514c763fbbad86d2337fde3377bf6c977365f0046576a85b94343`.
- Exact truth cache SHA-256: `005603c5a01c085762c9eecb266de6302f95356ca5eb4026f8c0380f57793f73`.
- Exact `m=49,k=92` episode/source archive SHA-256: `b0e410e9c0f5d0d9f0a4d3fbfd02cf247d171cd203c480083dc6fa666b68d928`.
- Exact fixed full-gene utility table SHA-256: `bd3b4f5097b4f158f58da102849d52080bb9ae7b7a4c4325f37bf574fd1a7c08`.
- Exact existing hierarchical bootstrap weights SHA-256: `f0a9697175fe2d4b8bc800e4352d961ed349f9e9a33df138c793be24f982c9cf`.

No predictor is trained, refit or inspected. No pathway, gene, episode, threshold, random projection or representation search may be added.

## Axis and episode definition

Map the fixed signed pathway vectors onto the exact ordered `G_PRIMARY` axis using the RESOLUTION-2 loader. Require the five mapped vectors to reproduce their fixed unit L2 norms and non-zero counts.

Use only the primary all-but-one episode at `m=49,k=92`. The fixed source tensor must have shape `1 × 50 × 49`, and each target's source set must equal the other 49 contexts exactly. For each plate, target and intervention:

\[
D_{t,p}=X_{t,p}-\frac{1}{49}\sum_{s\ne t}X_{s,p}
=\frac{50}{49}(X_{t,p}-\bar X_p).
\]

Plate 6 and Plate 14 remain separate until their cross product. The direct full-gene cross-product table, divided by `G=25,695` as in the fixed Tahoe metric, must reproduce the fixed `vtruth[m=49]` table within absolute tolerance `5×10^{-7}` per episode. The non-orthogonal five-score cross-product denominator must reproduce the existing fixed pooled pathway denominator within absolute tolerance `1×10^{-5}`.

## Pathway span and numerical tolerance

Let `W` be the `25,695 × 5` matrix of exact signed weights. Construct its compact SVD and use the retained left singular vectors `Q` as an implicit orthonormal projector `P_W=QQᵀ`, algebraically equivalent to `W(WᵀW)⁺Wᵀ`.

The rank tolerance is fixed as:

\[
\tau=s_{\max}\max(25{,}695,5)\epsilon_{64},
\qquad
r=\#\{s_i>\tau\}.
\]

The corresponding pseudoinverse relative cutoff is `max(25,695,5) × eps(float64)`. No outcome-dependent regularization is permitted. Report all singular values, the numerical rank, the condition number of `WᵀW`, and whether this fixed tolerance discards any direction.

## Primary fidelity

Project each plate truth separately using `Q`. With the exact fixed context × intervention aggregation:

\[
V_{\rm full}=\sum_{t,p}\langle D^{(6)}_{t,p},D^{(14)}_{t,p}\rangle,
\]

\[
V_{\rm path}=\sum_{t,p}\langle Q^\top D^{(6)}_{t,p},Q^\top D^{(14)}_{t,p}\rangle,
\]

\[
F_{\rm pathway}=V_{\rm path}/V_{\rm full}.
\]

The conventional `/G` normalization is reported and validated but cancels from the ratio. Reuse the already-fixed 10,000 context-then-intervention hierarchical bootstrap count matrix. Report the percentile 95% interval and bootstrap standard error for the single descriptive fidelity ratio; no threshold or multiplicity family is introduced.

## Secondary diagnostics

For each plate separately, pool the same truth episodes and report:

- squared-error fidelity `1-||D-P_WD||²/||D||²`;
- cosine between flattened `D` and `P_WD`;
- retained signal energy `||P_WD||²/||D||²`.

These are descriptive and secondary. Do not average plates before projection. Do not run random-projection controls.
