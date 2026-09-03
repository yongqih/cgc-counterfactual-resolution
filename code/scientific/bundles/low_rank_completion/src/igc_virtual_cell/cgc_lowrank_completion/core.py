from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


CONTEXTS = 50
INTERVENTIONS = 93
ENTRIES = CONTEXTS * INTERVENTIONS


def additive_design(indices: Iterable[int] | np.ndarray) -> np.ndarray:
    """Full-rank intercept + treatment-coded context/intervention design."""

    idx = np.asarray(indices, dtype=np.int64)
    context = idx // INTERVENTIONS
    intervention = idx % INTERVENTIONS
    design = np.zeros((len(idx), 1 + CONTEXTS - 1 + INTERVENTIONS - 1), dtype=np.float64)
    design[:, 0] = 1.0
    rows = np.arange(len(idx), dtype=np.int64)
    keep_context = context > 0
    keep_intervention = intervention > 0
    design[rows[keep_context], context[keep_context]] = 1.0
    design[rows[keep_intervention], CONTEXTS - 1 + intervention[keep_intervention]] = 1.0
    return design


def _xtx_inverse(design: np.ndarray) -> np.ndarray:
    gram = np.asarray(design.T @ design, dtype=np.float64)
    if np.linalg.matrix_rank(gram) != gram.shape[0]:
        raise RuntimeError("LOW_RANK_ADDITIVE_DESIGN_RANK_FAIL")
    return np.linalg.inv(gram)


@dataclass(frozen=True)
class FactorFit:
    rank: int
    ridge: float
    seed: int
    u: np.ndarray
    v: np.ndarray
    steps: int
    converged: bool
    objective_initial: float
    objective_final: float
    objective_best: float
    relative_change_window: float
    gradient_norm: float
    device: str
    peak_memory_bytes: int


@dataclass(frozen=True)
class PredictionWeights:
    additive: np.ndarray
    interaction: np.ndarray
    full: np.ndarray


@dataclass(frozen=True)
class ProfiledWorkspace:
    train_indices: np.ndarray
    k: object
    x: object
    xtx_inv: object
    context: object
    intervention: object
    energy_scale: object
    device: str


def build_profiled_workspace(
    gram: np.ndarray,
    train_indices: np.ndarray,
    *,
    device: str = "cuda",
) -> ProfiledWorkspace:
    """Materialize fold-local constants once for an entire candidate grid."""

    import torch

    train = np.asarray(train_indices, dtype=np.int64)
    if gram.shape != (len(train), len(train)):
        raise ValueError("gram must contain exactly the observed training rows")
    if not np.isfinite(gram).all():
        raise RuntimeError("LOW_RANK_NONFINITE_TRAIN_GRAM")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("LOW_RANK_CUDA_UNAVAILABLE")
    dtype = torch.float32
    dev = torch.device(device)
    k = torch.as_tensor(np.asarray(gram, dtype=np.float32), dtype=dtype, device=dev)
    design_np = additive_design(train)
    x = torch.as_tensor(design_np.astype(np.float32), dtype=dtype, device=dev)
    xtx_inv_np = _xtx_inverse(design_np)
    xtx_inv = torch.as_tensor(xtx_inv_np.astype(np.float32), dtype=dtype, device=dev)
    context = torch.as_tensor(train // INTERVENTIONS, dtype=torch.long, device=dev)
    intervention = torch.as_tensor(train % INTERVENTIONS, dtype=torch.long, device=dev)
    with torch.no_grad():
        xk = x.T @ k
        projected_trace = torch.trace((xtx_inv @ xk) @ x)
        residual_trace = torch.trace(k) - projected_trace
        energy_scale = torch.clamp(residual_trace / len(train), min=1e-12)
    return ProfiledWorkspace(train.copy(), k, x, xtx_inv, context, intervention, energy_scale, str(device))


def _center_factors(value):
    return value - value.mean(dim=0, keepdim=True)


def fit_profiled_cp(
    gram: np.ndarray,
    train_indices: np.ndarray,
    rank: int,
    ridge: float,
    seed: int,
    *,
    device: str = "cuda",
    max_steps: int = 120,
    min_steps: int = 30,
    convergence_window: int = 15,
    tolerance: float = 1e-5,
    gradient_tolerance: float = 1e-4,
    learning_rate: float = 0.05,
    workspace: ProfiledWorkspace | None = None,
) -> FactorFit:
    """Fit a profiled CP interaction using only an observed-row Gram matrix.

    The gene loadings are analytically profiled out. Additive main effects are
    removed by Frisch--Waugh--Lovell projection inside the observed set. The
    method never requires a held-out row or a Gram row/column touching it.
    """

    import torch

    if rank <= 0:
        raise ValueError("rank must be positive")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    train = np.asarray(train_indices, dtype=np.int64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("LOW_RANK_CUDA_UNAVAILABLE")
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats()
    dtype = torch.float32
    dev = torch.device(device)
    if workspace is None:
        workspace = build_profiled_workspace(gram, train, device=device)
    if workspace.device != str(device) or not np.array_equal(workspace.train_indices, train):
        raise RuntimeError("LOW_RANK_WORKSPACE_MISMATCH")
    k = workspace.k
    x = workspace.x
    xtx_inv = workspace.xtx_inv
    context = workspace.context
    intervention = workspace.intervention
    energy_scale = workspace.energy_scale

    generator = np.random.default_rng(int(seed))
    u0 = generator.normal(0.0, 0.5, size=(CONTEXTS, rank)).astype(np.float32)
    v0 = generator.normal(0.0, 0.5, size=(INTERVENTIONS, rank)).astype(np.float32)
    raw_u = torch.nn.Parameter(torch.as_tensor(u0, device=dev))
    raw_v = torch.nn.Parameter(torch.as_tensor(v0, device=dev))
    optimizer = torch.optim.Adam((raw_u, raw_v), lr=float(learning_rate))

    history: list[float] = []
    best = float("inf")
    last_grad = float("inf")

    def objective():
        u = _center_factors(raw_u)
        v = _center_factors(raw_v)
        z = u[context] * v[intervention]
        projection = xtx_inv @ (x.T @ z)
        zp = z - x @ projection
        kz = k @ zp
        kres_z = kz - x @ (xtx_inv @ (x.T @ kz))
        normal = zp.T @ zp + float(ridge) * len(train) * torch.eye(rank, device=dev, dtype=dtype)
        signal = zp.T @ kres_z
        explained = torch.trace(torch.linalg.solve(normal, signal)) / (energy_scale * len(train))
        factor_penalty = float(ridge) * (torch.mean(u.square()) + torch.mean(v.square()))
        return -explained + factor_penalty

    converged = False
    for step in range(1, int(max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("LOW_RANK_NONFINITE_OBJECTIVE")
        loss.backward()
        last_grad = float(
            torch.sqrt(sum(torch.sum(parameter.grad.square()) for parameter in (raw_u, raw_v))).detach().cpu()
        )
        torch.nn.utils.clip_grad_norm_((raw_u, raw_v), max_norm=10.0)
        optimizer.step()
        # Exact gauge balancing leaves every u_c*r v_p*r product unchanged and
        # minimizes the paired factor penalty for the current component product.
        with torch.no_grad():
            uc = _center_factors(raw_u)
            vc = _center_factors(raw_v)
            un = torch.sqrt(torch.mean(uc.square(), dim=0).clamp_min(1e-12))
            vn = torch.sqrt(torch.mean(vc.square(), dim=0).clamp_min(1e-12))
            balance = torch.sqrt(vn / un)
            raw_u.mul_(balance)
            raw_v.div_(balance)
        value = float(loss.detach().cpu())
        history.append(value)
        best = min(best, value)
        if step >= max(min_steps, convergence_window + 1):
            recent = np.asarray(history[-convergence_window:], dtype=np.float64)
            denom = max(abs(float(recent[0])), 1e-12)
            relative = abs(float(recent[-1] - recent[0])) / denom
            if relative <= tolerance or last_grad <= gradient_tolerance:
                converged = True
                break

    recent = np.asarray(history[-min(convergence_window, len(history)):], dtype=np.float64)
    relative = (
        abs(float(recent[-1] - recent[0])) / max(abs(float(recent[0])), 1e-12)
        if len(recent) > 1
        else float("inf")
    )
    with torch.no_grad():
        u = _center_factors(raw_u).detach().cpu().numpy().astype(np.float64)
        v = _center_factors(raw_v).detach().cpu().numpy().astype(np.float64)
    peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
    return FactorFit(
        rank=int(rank),
        ridge=float(ridge),
        seed=int(seed),
        u=u,
        v=v,
        steps=len(history),
        converged=bool(converged),
        objective_initial=float(history[0]),
        objective_final=float(history[-1]),
        objective_best=float(best),
        relative_change_window=float(relative),
        gradient_norm=float(last_grad),
        device=str(device),
        peak_memory_bytes=peak,
    )


def prediction_weights(
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    fit: FactorFit | None,
) -> PredictionWeights:
    """Return observed-row coefficients for additive and full predictions."""

    train = np.asarray(train_indices, dtype=np.int64)
    predict = np.asarray(predict_indices, dtype=np.int64)
    x_train = additive_design(train)
    x_predict = additive_design(predict)
    xtx_inv = _xtx_inverse(x_train)
    additive = np.asarray(x_predict @ xtx_inv @ x_train.T, dtype=np.float64)
    if fit is None:
        interaction = np.zeros_like(additive)
        return PredictionWeights(additive, interaction, additive.copy())

    z_train = fit.u[train // INTERVENTIONS] * fit.v[train % INTERVENTIONS]
    z_predict = fit.u[predict // INTERVENTIONS] * fit.v[predict % INTERVENTIONS]
    projection = xtx_inv @ x_train.T @ z_train
    zp = z_train - x_train @ projection
    hp = z_predict - x_predict @ projection
    normal = zp.T @ zp + fit.ridge * len(train) * np.eye(fit.rank, dtype=np.float64)
    interaction = np.asarray(hp @ np.linalg.solve(normal, zp.T), dtype=np.float64)
    return PredictionWeights(additive, interaction, additive + interaction)


def weighted_squared_error(
    full_gram: np.ndarray,
    train_indices: np.ndarray,
    target_indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Per-target squared error from a full sample Gram matrix."""

    train = np.asarray(train_indices, dtype=np.int64)
    target = np.asarray(target_indices, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (len(target), len(train)):
        raise ValueError("weight shape mismatch")
    gram_tt = np.asarray(full_gram[np.ix_(train, train)], dtype=np.float64)
    gram_tv = np.asarray(full_gram[np.ix_(train, target)], dtype=np.float64)
    truth = np.asarray(np.diag(full_gram[np.ix_(target, target)]), dtype=np.float64)
    pred_cross = np.einsum("ij,ji->i", w, gram_tv, optimize=True)
    pred_norm = np.einsum("ij,jk,ik->i", w, gram_tt, w, optimize=True)
    return truth - 2.0 * pred_cross + pred_norm
