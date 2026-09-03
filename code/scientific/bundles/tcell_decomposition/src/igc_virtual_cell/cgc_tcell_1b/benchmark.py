"""Frozen CGC-TCELL-1B strict-LODO training and geometry evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
import yaml

from igc_virtual_cell.cgc_tcell.core import hash_block
from igc_virtual_cell.cgc_tcell_1b.audit import git, load_frozen, sha256_file


STATE_PAIRS = ((0, 1), (0, 2), (1, 2))
STATE_PAIR_NAMES = ("Rest__Stim8hr", "Rest__Stim48hr", "Stim8hr__Stim48hr")
TRAINED_MODELS = ("Ridge", "Bilinear", "MLP")


def two_way(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    return (
        x
        - x.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
        - x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        + x.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    )


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _iter_target_blocks(delta: np.ndarray, donors: list[int], batch: int):
    for start in range(0, delta.shape[0], batch):
        stop = min(start + batch, delta.shape[0])
        block = np.asarray(delta[start:stop, donors, :, :], dtype=np.float32)
        yield start, stop, block.reshape(-1, delta.shape[-1])


def fit_response_basis(
    delta: np.ndarray,
    donors: list[int],
    *,
    rank: int,
    oversample: int,
    batch_targets: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Two-pass streamed randomized PCA using outer-training responses only."""
    started = time.perf_counter()
    genes = delta.shape[-1]
    rows = delta.shape[0] * len(donors) * delta.shape[2]
    mean = np.zeros(genes, dtype=np.float64)
    total_sq = 0.0
    for _, _, block in _iter_target_blocks(delta, donors, batch_targets):
        mean += block.sum(axis=0, dtype=np.float64)
    mean /= rows
    q = min(rank + oversample, genes, rows)
    generator = torch.Generator(device=device).manual_seed(seed)
    omega = torch.randn(genes, q, generator=generator, device=device, dtype=torch.float32)
    y = torch.empty((rows, q), dtype=torch.float32, device=device)
    cursor = 0
    mean_t = torch.from_numpy(mean.astype(np.float32)).to(device)
    for _, _, block in _iter_target_blocks(delta, donors, batch_targets):
        a = torch.from_numpy(block).to(device)
        a = a - mean_t
        count = len(block)
        y[cursor : cursor + count] = a @ omega
        total_sq += float(torch.sum(a.double() * a.double()).item())
        cursor += count
    qmat = torch.linalg.qr(y, mode="reduced").Q
    del y, omega
    b = torch.zeros((q, genes), dtype=torch.float32, device=device)
    cursor = 0
    for _, _, block in _iter_target_blocks(delta, donors, batch_targets):
        a = torch.from_numpy(block).to(device) - mean_t
        count = len(block)
        b += qmat[cursor : cursor + count].T @ a
        cursor += count
    _, singular, vh = torch.linalg.svd(b, full_matrices=False)
    components = vh[:rank].contiguous().cpu().numpy().astype(np.float32)
    captured = float(torch.sum(singular[:rank].double() ** 2).item())
    del qmat, b, singular, vh, mean_t
    torch.cuda.empty_cache()
    return components, mean.astype(np.float32), {
        "training_rows": rows,
        "rank": rank,
        "captured_training_variance_fraction": captured / total_sq,
        "fit_seconds": time.perf_counter() - started,
    }


def project_responses(
    delta: np.ndarray,
    components: np.ndarray,
    path: Path,
    *,
    donors: list[int],
    batch_targets: int,
    device: torch.device,
) -> np.memmap:
    projected = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(*delta.shape[:3], components.shape[0]),
    )
    projected[:] = np.nan
    project_response_donors(
        delta,
        components,
        projected,
        donors=donors,
        batch_targets=batch_targets,
        device=device,
    )
    projected.flush()
    return projected


def project_response_donors(
    delta: np.ndarray,
    components: np.ndarray,
    projected: np.ndarray,
    *,
    donors: list[int],
    batch_targets: int,
    device: torch.device,
) -> None:
    basis = torch.from_numpy(components.T.copy()).to(device)
    for start in range(0, delta.shape[0], batch_targets):
        stop = min(start + batch_targets, delta.shape[0])
        block = np.asarray(delta[start:stop, donors], dtype=np.float32)
        flat = torch.from_numpy(block.reshape(-1, delta.shape[-1])).to(device)
        projected[start:stop, donors] = (flat @ basis).cpu().numpy().reshape(
            stop - start, len(donors), delta.shape[2], -1
        )
    projected.flush()
    del basis
    torch.cuda.empty_cache()


def context_features(ntc: np.ndarray, train: list[int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train_values = ntc[train].reshape(-1, ntc.shape[-1]).astype(np.float64)
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0, ddof=0)
    scale[scale <= 1e-12] = 1.0
    standardized = (train_values - mean) / scale
    _, singular, vh = np.linalg.svd(standardized, full_matrices=False)
    available = int(np.sum(singular > singular[0] * 1e-10)) if singular[0] else 0
    dim = min(8, available)
    components = vh[:dim]
    all_features = ((ntc.reshape(-1, ntc.shape[-1]) - mean) / scale) @ components.T
    return all_features.reshape(4, 3, dim).astype(np.float32), {
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "components": components.astype(np.float32),
        "singular_values": singular[:dim].astype(np.float32),
        "training_donor_indices": np.asarray(train, dtype=np.int64),
    }


def _anchor(coeff: np.ndarray, donors: list[int], blind: bool = False) -> np.ndarray:
    values = coeff[:, donors].mean(axis=1, dtype=np.float64).astype(np.float32)
    if blind:
        pooled = values.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
        values = np.repeat(pooled, 3, axis=1)
    return values


def select_ridge_alpha(
    coeff: np.ndarray,
    h: np.ndarray,
    outer_train: list[int],
    alphas: list[float],
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for valid in outer_train:
        train = [d for d in outer_train if d != valid]
        anchor = _anchor(coeff, train)
        y = coeff[:, train].reshape(coeff.shape[0], -1, coeff.shape[-1])
        states = np.tile(np.arange(3), len(train))
        y = y - anchor[:, states]
        x = h[train].reshape(-1, h.shape[-1]).astype(np.float64)
        xv = h[valid].astype(np.float64)
        truth = coeff[:, valid]
        for alpha in alphas:
            inverse = np.linalg.inv(x.T @ x + float(alpha) * np.eye(x.shape[1]))
            projection = inverse @ x.T
            weights = np.einsum("kn,pnr->pkr", projection, y, optimize=True)
            prediction = anchor + np.einsum("sk,pkr->psr", xv, weights, optimize=True)
            mse = float(np.mean(np.square(prediction - truth, dtype=np.float64)))
            rows.append(
                {
                    "model": "Ridge",
                    "inner_validation_donor_index": valid,
                    "alpha": alpha,
                    "seed": -1,
                    "best_epoch": 0,
                    "validation_response_mse_rank256": mse,
                }
            )
    frame = pd.DataFrame(rows)
    selected = float(frame.groupby("alpha")["validation_response_mse_rank256"].mean().idxmin())
    return selected, rows


def fit_ridge_predict(
    coeff: np.ndarray,
    h: np.ndarray,
    train: list[int],
    held: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    anchor = _anchor(coeff, train)
    y = coeff[:, train].reshape(coeff.shape[0], -1, coeff.shape[-1])
    states = np.tile(np.arange(3), len(train))
    y = y - anchor[:, states]
    x = h[train].reshape(-1, h.shape[-1]).astype(np.float64)
    inverse = np.linalg.inv(x.T @ x + alpha * np.eye(x.shape[1]))
    weights = np.einsum("kn,pnr->pkr", inverse @ x.T, y, optimize=True).astype(np.float32)
    residual = np.einsum("sk,pkr->psr", h[held], weights, optimize=True).astype(np.float32)
    return residual, weights


class Bilinear(nn.Module):
    def __init__(self, perturbations: int, context_dim: int, output_dim: int) -> None:
        super().__init__()
        self.intervention = nn.Embedding(perturbations, 32)
        self.context = nn.Linear(context_dim, 32)
        self.output = nn.Linear(32, output_dim)

    def forward(self, p: torch.Tensor, h: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del state
        return self.output(self.intervention(p) * self.context(h))


class ContextMLP(nn.Module):
    def __init__(self, perturbations: int, context_dim: int, output_dim: int) -> None:
        super().__init__()
        self.intervention = nn.Embedding(perturbations, 32)
        self.network = nn.Sequential(
            nn.Linear(32 + context_dim + 3, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(self, p: torch.Tensor, h: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(state, num_classes=3).to(h.dtype)
        return self.network(torch.cat((self.intervention(p), h, onehot), dim=1))


def _neural_arrays(
    coeff: np.ndarray, h: np.ndarray, donors: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor = _anchor(coeff, donors)
    p = np.repeat(np.arange(coeff.shape[0]), len(donors) * 3)
    d = np.tile(np.repeat(np.asarray(donors), 3), coeff.shape[0])
    s = np.tile(np.arange(3), coeff.shape[0] * len(donors))
    target = coeff[:, donors].reshape(-1, coeff.shape[-1]) - anchor[p, s]
    return p.astype(np.int64), h[d, s], s.astype(np.int64), target.astype(np.float32), anchor


def train_neural(
    model_name: str,
    coeff: np.ndarray,
    h: np.ndarray,
    train_donors: list[int],
    valid_donor: int | None,
    *,
    learning_rate: float,
    seed: int,
    config: dict[str, Any],
    fixed_epochs: int | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    _seed(seed)
    device = torch.device("cuda")
    cls = Bilinear if model_name == "Bilinear" else ContextMLP
    model = cls(coeff.shape[0], h.shape[-1], coeff.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=float(config["weight_decay"])
    )
    p, context, state, target, _ = _neural_arrays(coeff, h, train_donors)
    p_t = torch.from_numpy(p).to(device)
    h_t = torch.from_numpy(context).to(device)
    s_t = torch.from_numpy(state).to(device)
    y_t = torch.from_numpy(target).to(device)
    if valid_donor is not None:
        anchor = _anchor(coeff, train_donors)
        vp = np.repeat(np.arange(coeff.shape[0]), 3)
        vs = np.tile(np.arange(3), coeff.shape[0])
        vh = h[valid_donor, vs]
        vy = coeff[:, valid_donor].reshape(-1, coeff.shape[-1]) - anchor[vp, vs]
        vp_t = torch.from_numpy(vp.astype(np.int64)).to(device)
        vs_t = torch.from_numpy(vs.astype(np.int64)).to(device)
        vh_t = torch.from_numpy(vh.astype(np.float32)).to(device)
        vy_t = torch.from_numpy(vy.astype(np.float32)).to(device)
    max_epochs = fixed_epochs or int(config["max_epochs"])
    patience = int(config["early_stopping_patience"])
    batch = int(config["batch_size"])
    best_mse = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        generator = torch.Generator(device=device).manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(p_t), generator=generator, device=device)
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(p_t[idx], h_t[idx], s_t[idx])
            loss = torch.mean((prediction - y_t[idx]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
        if valid_donor is None:
            continue
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(vp_t), batch):
                stop = min(start + batch, len(vp_t))
                pred = model(vp_t[start:stop], vh_t[start:stop], vs_t[start:stop])
                total += float(torch.sum((pred - vy_t[start:stop]) ** 2).item())
                count += pred.numel()
        mse = total / count
        if mse < best_mse - 1e-10:
            best_mse = mse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if valid_donor is not None and best_state is not None:
        model.load_state_dict(best_state)
    info = {
        "best_validation_mse": None if valid_donor is None else best_mse,
        "best_epoch": max_epochs if valid_donor is None else best_epoch,
        "epochs_run": epoch,
        "seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    del p_t, h_t, s_t, y_t
    if valid_donor is not None:
        del vp_t, vs_t, vh_t, vy_t
    return model, info


def predict_neural(
    model: nn.Module, h: np.ndarray, held: int, perturbations: int, batch: int
) -> np.ndarray:
    device = torch.device("cuda")
    p = np.repeat(np.arange(perturbations), 3).astype(np.int64)
    s = np.tile(np.arange(3), perturbations).astype(np.int64)
    context = h[held, s].astype(np.float32)
    output = np.empty((len(p), model.output.out_features if isinstance(model, Bilinear) else 256), np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(p), batch):
            stop = min(start + batch, len(p))
            pred = model(
                torch.from_numpy(p[start:stop]).to(device),
                torch.from_numpy(context[start:stop]).to(device),
                torch.from_numpy(s[start:stop]).to(device),
            )
            output[start:stop] = pred.cpu().numpy()
    return output.reshape(perturbations, 3, -1)


class PredictionWriter:
    def __init__(self, path: Path, rank: int) -> None:
        self.path = path
        self.rank = rank
        self.writer: pq.ParquetWriter | None = None

    def append(
        self,
        values: np.ndarray,
        *,
        targets: list[str],
        states: list[str],
        outer_fold: int,
        held_donor: str,
        model: str,
        seed: int,
        anchor: str,
    ) -> None:
        flat = np.ascontiguousarray(values.reshape(-1, self.rank), dtype=np.float32)
        coeff = pa.FixedSizeListArray.from_arrays(pa.array(flat.ravel()), self.rank)
        n = len(flat)
        table = pa.table(
            {
                "outer_fold": pa.array(np.full(n, outer_fold, np.int16)),
                "held_out_donor": pa.array([held_donor] * n),
                "model": pa.array([model] * n),
                "seed": pa.array(np.full(n, seed, np.int16)),
                "anchor": pa.array([anchor] * n),
                "perturbation_id": pa.array(np.repeat(targets, 3)),
                "stimulation_state": pa.array(np.tile(states, len(targets))),
                "residual_response_pca_coefficients": coeff,
            }
        )
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table, row_group_size=8192)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _corr(sum_x: float, sum_y: float, x2: float, y2: float, xy: float, n: int) -> float:
    cov = xy - sum_x * sum_y / n
    vx = x2 - sum_x * sum_x / n
    vy = y2 - sum_y * sum_y / n
    return float(cov / math.sqrt(max(vx * vy, 1e-30)))


def _cos(xy: float, x2: float, y2: float) -> float:
    return float(xy / math.sqrt(max(x2 * y2, 1e-30)))


class FoldEvaluator:
    """Precompute exact gene-space sufficient statistics for one outer fold."""

    def __init__(
        self,
        delta: np.ndarray,
        coeff: np.ndarray,
        components: np.ndarray,
        ntc: np.ndarray,
        train: list[int],
        held: int,
        gene_blocks: np.ndarray,
        chunk: int,
    ) -> None:
        self.delta, self.coeff, self.v = delta, coeff, components
        self.ntc, self.train, self.held = ntc, train, held
        self.gene_blocks, self.chunk = gene_blocks, chunk
        self.truth_coeff = np.asarray(coeff[:, held], np.float32)
        self.anchor_coeff = _anchor(coeff, train)
        self.blind_coeff = _anchor(coeff, train, blind=True)
        self.operator: dict[str, Any] = {}
        self.base: dict[str, Any] = {}
        self._precompute()

    def _blank_operator(self) -> dict[str, Any]:
        p = self.delta.shape[0]
        r = self.v.shape[0]
        return {
            "truth2": np.zeros(p), "anchor2": np.zeros(p), "dot": np.zeros(p),
            "truth_proj": np.zeros((p, 3, r), np.float32),
            "anchor_proj": np.zeros((p, 3, r), np.float32),
            "contrast_truth2": np.zeros((p, 3)),
            "contrast_anchor2": np.zeros((p, 3)),
            "contrast_dot": np.zeros((p, 3)),
            "contrast_truth_proj": np.zeros((p, 3, r), np.float32),
            "contrast_anchor_proj": np.zeros((p, 3, r), np.float32),
        }

    @staticmethod
    def _blank_base() -> dict[str, float]:
        return {key: 0.0 for key in ("sum_truth", "sum_base", "truth2", "base2", "dot")}

    def _precompute(self) -> None:
        self.operator["total"] = self._blank_operator()
        for block in range(5):
            self.operator[f"block_{block}"] = self._blank_operator()
        self.base = {
            "response_anchor": self._blank_base(), "response_blind": self._blank_base(),
            "state_anchor": self._blank_base(), "state_blind": self._blank_base(),
        }
        self.vsum = self.v.sum(axis=1, dtype=np.float64)
        self.gram = {"total": np.eye(self.v.shape[0], dtype=np.float64)}
        for block in range(5):
            cols = self.gene_blocks == block
            vb = self.v[:, cols].astype(np.float64)
            self.gram[f"block_{block}"] = vb @ vb.T
        p, _, genes = self.delta.shape[0], 3, self.delta.shape[-1]
        for start in range(0, genes, self.chunk):
            stop = min(start + self.chunk, genes)
            truth_delta = np.asarray(self.delta[:, self.held, :, start:stop], np.float32)
            anchor = np.asarray(self.delta[:, self.train, :, start:stop], np.float32).mean(
                axis=1, dtype=np.float64
            ).astype(np.float32)
            blind = anchor.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
            blind = np.repeat(blind, 3, axis=1)
            truth_op, anchor_op = two_way(truth_delta), two_way(anchor)
            ntc = self.ntc[self.held, :, start:stop][None, :, :]
            for key, x, a in (
                ("response_anchor", truth_delta, anchor),
                ("response_blind", truth_delta, blind),
                ("state_anchor", truth_delta + ntc, anchor + ntc),
                ("state_blind", truth_delta + ntc, blind + ntc),
            ):
                item = self.base[key]
                item["sum_truth"] += float(x.sum(dtype=np.float64))
                item["sum_base"] += float(a.sum(dtype=np.float64))
                item["truth2"] += float(np.square(x, dtype=np.float64).sum())
                item["base2"] += float(np.square(a, dtype=np.float64).sum())
                item["dot"] += float(np.multiply(x, a, dtype=np.float64).sum())
            for label, mask in [("total", np.ones(stop - start, bool))] + [
                (f"block_{b}", self.gene_blocks[start:stop] == b) for b in range(5)
            ]:
                if not mask.any():
                    continue
                t, a, vb = truth_op[:, :, mask], anchor_op[:, :, mask], self.v[:, start:stop][:, mask]
                item = self.operator[label]
                item["truth2"] += np.square(t, dtype=np.float64).sum(axis=(1, 2))
                item["anchor2"] += np.square(a, dtype=np.float64).sum(axis=(1, 2))
                item["dot"] += np.multiply(t, a, dtype=np.float64).sum(axis=(1, 2))
                item["truth_proj"] += np.einsum("psg,rg->psr", t, vb, optimize=True)
                item["anchor_proj"] += np.einsum("psg,rg->psr", a, vb, optimize=True)
                for pair_index, (left, right) in enumerate(STATE_PAIRS):
                    tc, ac = t[:, left] - t[:, right], a[:, left] - a[:, right]
                    item["contrast_truth2"][:, pair_index] += np.square(tc, dtype=np.float64).sum(axis=1)
                    item["contrast_anchor2"][:, pair_index] += np.square(ac, dtype=np.float64).sum(axis=1)
                    item["contrast_dot"][:, pair_index] += np.multiply(tc, ac, dtype=np.float64).sum(axis=1)
                    item["contrast_truth_proj"][:, pair_index] += tc @ vb.T
                    item["contrast_anchor_proj"][:, pair_index] += ac @ vb.T
        self.n = p * 3 * genes
        ntc_proj = self.ntc[self.held] @ self.v.T
        self.projections = {
            "response_truth": self.truth_coeff,
            "response_anchor": self.anchor_coeff,
            "response_blind": self.blind_coeff,
            "state_truth": self.truth_coeff + ntc_proj[None],
            "state_anchor": self.anchor_coeff + ntc_proj[None],
            "state_blind": self.blind_coeff + ntc_proj[None],
        }

    def evaluate(
        self, residual: np.ndarray, *, anchor_kind: str
    ) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
        residual = np.asarray(residual, np.float32)
        rgamma = two_way(residual)
        op = self.operator["total"]
        blind = anchor_kind == "CONTEXT_BLIND_ANCHOR"
        anchor2 = np.zeros_like(op["anchor2"]) if blind else op["anchor2"]
        anchor_dot = np.zeros_like(op["dot"]) if blind else op["dot"]
        anchor_proj = np.zeros_like(op["anchor_proj"]) if blind else op["anchor_proj"]
        pred2 = anchor2 + 2 * np.einsum("psr,psr->p", anchor_proj, rgamma)
        # PCA component rows are orthonormal.  In the complete strict-trans
        # space this quadratic form is therefore exactly ||r_gamma||^2 (up to
        # the floating-point QR/SVD tolerance), avoiding a repeated 256x256
        # product for every intervention and state.
        pred2 += np.square(rgamma, dtype=np.float64).sum(axis=(1, 2))
        dot = anchor_dot + np.einsum("psr,psr->p", op["truth_proj"], rgamma)
        truth2 = op["truth2"]
        contrast_pred2 = np.empty((len(truth2), 3))
        contrast_dot = np.empty_like(contrast_pred2)
        for i, (left, right) in enumerate(STATE_PAIRS):
            rc = rgamma[:, left] - rgamma[:, right]
            contrast_pred2[:, i] = 0.0 if blind else op["contrast_anchor2"][:, i]
            if not blind:
                contrast_pred2[:, i] += 2 * np.einsum("pr,pr->p", op["contrast_anchor_proj"][:, i], rc)
            contrast_pred2[:, i] += np.square(rc, dtype=np.float64).sum(axis=1)
            contrast_dot[:, i] = (0.0 if blind else op["contrast_dot"][:, i]) + np.einsum(
                "pr,pr->p", op["contrast_truth_proj"][:, i], rc
            )
        overall = {
            "truth2": float(truth2.sum()), "pred2": float(pred2.sum()), "dot": float(dot.sum()),
            "kappa_c": float(pred2.sum() / truth2.sum()),
            "alpha": float(dot.sum() / truth2.sum()),
            "cosine": np.nan if pred2.sum() <= 1e-30 else _cos(float(dot.sum()), float(truth2.sum()), float(pred2.sum())),
            "pearson": np.nan if pred2.sum() <= 1e-30 else _cos(float(dot.sum()), float(truth2.sum()), float(pred2.sum())),
        }
        distances = []
        for i, name in enumerate(STATE_PAIR_NAMES):
            t2, p2, dp = op["contrast_truth2"][:, i].sum(), contrast_pred2[:, i].sum(), contrast_dot[:, i].sum()
            distances.append(
                {
                    "contrast": name, "d2_true": float(t2 / (len(truth2) * self.delta.shape[-1])),
                    "d2_pred": float(p2 / (len(truth2) * self.delta.shape[-1])),
                    "retention_ratio": float(p2 / t2), "cosine": np.nan if p2 <= 1e-30 else _cos(float(dp), float(t2), float(p2)),
                    "pearson": np.nan if p2 <= 1e-30 else _cos(float(dp), float(t2), float(p2)),
                }
            )
        per_p = {
            "truth2": truth2, "pred2": pred2, "dot": dot,
            "contrast_truth2": op["contrast_truth2"], "contrast_pred2": contrast_pred2,
        }
        block_rows = []
        total_raw = {
            "truth2": float(truth2.sum()),
            "pred2": float(pred2.sum()),
            "dot": float(dot.sum()),
            "contrast_truth2": float(op["contrast_truth2"][:, 1].sum()),
            "contrast_pred2": float(contrast_pred2[:, 1].sum()),
        }
        for block in range(5):
            raw = self._gene_block_raw(rgamma, f"block_{block}", blind)
            block_rows.append(self._raw_block_metrics(raw, block, "block_only"))
            complement = {key: total_raw[key] - raw[key] for key in total_raw}
            block_rows.append(
                self._raw_block_metrics(complement, block, "leave_one_block_out")
            )
        base_name = "anchor" if anchor_kind == "MEAN_STATE_ANCHOR" else "blind"
        base_response = self.base[f"response_{base_name}"]
        base_state = self.base[f"state_{base_name}"]
        rflat = residual.reshape(-1, residual.shape[-1]).astype(np.float64)
        r2 = float(np.square(rflat).sum())
        rsum = float((rflat @ self.vsum).sum())
        response = self._base_metrics(base_response, self.projections[f"response_truth"], self.projections[f"response_{base_name}"], rflat, r2, rsum)
        state = self._base_metrics(base_state, self.projections[f"state_truth"], self.projections[f"state_{base_name}"], rflat, r2, rsum)
        overall["response_metrics"] = response
        overall["state_metrics"] = state
        overall["distances"] = distances
        return overall, per_p, block_rows

    def _base_metrics(self, base, truth_proj, base_proj, rflat, r2, rsum):
        tp, bp = truth_proj.reshape(-1, truth_proj.shape[-1]), base_proj.reshape(-1, base_proj.shape[-1])
        dot = base["dot"] + float(np.multiply(tp, rflat).sum())
        pred2 = base["base2"] + 2 * float(np.multiply(bp, rflat).sum()) + r2
        sump = base["sum_base"] + rsum
        mse = (base["truth2"] + pred2 - 2 * dot) / self.n
        return {
            "pearson": _corr(base["sum_truth"], sump, base["truth2"], pred2, dot, self.n),
            "cosine": _cos(dot, base["truth2"], pred2), "mse": float(mse),
        }

    @staticmethod
    def _quadratic(values: np.ndarray, gram: np.ndarray) -> float:
        if values.size < 100_000 or not torch.cuda.is_available():
            return float(np.einsum("...r,rt,...t->", values, gram, values, optimize=True))
        x = torch.from_numpy(np.ascontiguousarray(values)).to("cuda")
        g = torch.from_numpy(np.asarray(gram, dtype=np.float32)).to("cuda")
        result = float(torch.sum((x @ g) * x, dtype=torch.float64).item())
        del x, g
        return result

    def _gene_block_raw(self, rgamma, label, blind):
        item = self.operator[label]
        truth2 = float(item["truth2"].sum())
        anchor2 = float(item["anchor2"].sum())
        dot0 = float(item["dot"].sum())
        aproj = item["anchor_proj"]
        ca = float(item["contrast_anchor2"][:, 1].sum())
        cd = float(item["contrast_dot"][:, 1].sum())
        cap = item["contrast_anchor_proj"][:, 1]
        if blind:
            anchor2 = dot0 = ca = cd = 0.0
        pred2 = anchor2
        if not blind:
            pred2 += 2 * float(np.einsum("psr,psr->", aproj, rgamma))
        pred2 += self._quadratic(rgamma, self.gram[label])
        dot = dot0 + float(np.einsum("psr,psr->", item["truth_proj"], rgamma))
        rc = rgamma[:, 0] - rgamma[:, 2]
        cp = ca
        if not blind:
            cp += 2 * float(np.einsum("pr,pr->", cap, rc))
        cp += self._quadratic(rc, self.gram[label])
        return {
            "truth2": truth2,
            "pred2": pred2,
            "dot": dot,
            "contrast_truth2": float(item["contrast_truth2"][:, 1].sum()),
            "contrast_pred2": cp,
        }

    @staticmethod
    def _raw_block_metrics(raw, block, analysis):
        return {
            "analysis": analysis,
            "block": block,
            "kappa_c": raw["pred2"] / raw["truth2"],
            "alpha": raw["dot"] / raw["truth2"],
            "cosine": np.nan if raw["pred2"] <= 1e-30 else _cos(raw["dot"], raw["truth2"], raw["pred2"]),
            "rest_stim48_retention": raw["contrast_pred2"] / raw["contrast_truth2"],
        }


def intervention_block_rows(per_p: dict[str, np.ndarray], targets: list[str]) -> list[dict[str, Any]]:
    assignment = np.asarray([hash_block(target) for target in targets])
    rows = []
    for block in range(5):
        for analysis, keep in (("block_only", assignment == block), ("leave_one_block_out", assignment != block)):
            t2, p2, dot = (per_p[key][keep].sum() for key in ("truth2", "pred2", "dot"))
            ct = per_p["contrast_truth2"][keep, 1].sum(); cp = per_p["contrast_pred2"][keep, 1].sum()
            rows.append({"analysis": analysis, "block": block, "kappa_c": p2/t2, "alpha": dot/t2, "cosine": _cos(dot,t2,p2), "rest_stim48_retention": cp/ct})
    return rows


def bootstrap_model(
    sufficient: dict[int, list[dict[str, np.ndarray]]], draws: int, seed: int
) -> list[dict[str, Any]]:
    seeds = sorted(sufficient)
    p = len(sufficient[seeds[0]][0]["truth2"])
    rng = np.random.default_rng(seed)
    metric_names = ["kappa_c", "alpha", "cosine", "pearson", *[f"R_{x}" for x in STATE_PAIR_NAMES]]
    values = {name: np.empty(draws) for name in metric_names}
    exact = {name: [] for name in metric_names}
    for model_seed in seeds:
        donors = sufficient[model_seed]
        t2 = sum(float(d["truth2"].sum()) for d in donors)
        p2 = sum(float(d["pred2"].sum()) for d in donors)
        dot = sum(float(d["dot"].sum()) for d in donors)
        exact["kappa_c"].append(p2 / t2)
        exact["alpha"].append(dot / t2)
        direction = np.nan if p2 <= 1e-30 else dot / np.sqrt(t2 * p2)
        exact["cosine"].append(direction)
        exact["pearson"].append(direction)
        for ci, name in enumerate(STATE_PAIR_NAMES):
            ct = sum(float(d["contrast_truth2"][:, ci].sum()) for d in donors)
            cp = sum(float(d["contrast_pred2"][:, ci].sum()) for d in donors)
            exact[f"R_{name}"].append(cp / ct)
    batch = 32
    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        indices = rng.integers(0, p, size=(stop-start, p), dtype=np.int32)
        per_seed = {name: [] for name in values}
        for model_seed in seeds:
            donors = sufficient[model_seed]
            t2 = np.stack([sum(d["truth2"][idx].sum() for d in donors) for idx in indices])
            p2 = np.stack([sum(d["pred2"][idx].sum() for d in donors) for idx in indices])
            dot = np.stack([sum(d["dot"][idx].sum() for d in donors) for idx in indices])
            per_seed["kappa_c"].append(p2/t2); per_seed["alpha"].append(dot/t2)
            denominator = np.sqrt(t2 * p2)
            direction = np.divide(dot, denominator, out=np.full_like(dot, np.nan), where=denominator > 1e-30)
            per_seed["cosine"].append(direction); per_seed["pearson"].append(direction)
            for ci,name in enumerate(STATE_PAIR_NAMES):
                ct=np.stack([sum(d["contrast_truth2"][idx,ci].sum() for d in donors) for idx in indices])
                cp=np.stack([sum(d["contrast_pred2"][idx,ci].sum() for d in donors) for idx in indices])
                per_seed[f"R_{name}"].append(cp/ct)
        for name in values:
            values[name][start:stop] = np.mean(per_seed[name], axis=0)
    rows=[]
    for name,array in values.items():
        exact_values = np.asarray(exact[name], dtype=np.float64)
        if np.isnan(exact_values).all():
            point = lower = upper = np.nan
        else:
            point = float(np.nanmean(exact_values))
            lower = float(np.nanquantile(array, .025))
            upper = float(np.nanquantile(array, .975))
        rows.append({"metric":name,"point":point,"ci_lower":lower,"ci_upper":upper,"draws":draws})
    return rows


def _classification(model: str, donor: pd.DataFrame, boot: pd.DataFrame, fidelity: pd.DataFrame) -> tuple[str,str,dict[str,Any]]:
    b=boot.set_index("metric")
    alpha=float(b.loc["alpha","point"]); upper=float(b.loc["alpha","ci_upper"])
    donor_alpha=donor.groupby("held_out_donor")["alpha"].mean()
    if alpha<.5 and upper<.75 and int((donor_alpha<.5).sum())>=3: amplitude="CONTEXT_AMPLITUDE_COMPRESSION_STRONG"
    elif alpha<.8: amplitude="CONTEXT_AMPLITUDE_COMPRESSION_MODERATE"
    elif float(b.loc["alpha","ci_lower"])<=1<=upper or .8<=alpha<=1.2: amplitude="CONTEXT_AMPLITUDE_PRESERVED"
    else: amplitude="CONTEXT_AMPLITUDE_COMPRESSION_NOT_ESTABLISHED"
    cosine_low=float(b.loc["cosine","ci_lower"]); pearson_low=float(b.loc["pearson","ci_lower"])
    top=int(donor.groupby("held_out_donor")["top_contrast_correct"].mean().gt(.5).sum())
    med=float(fidelity.groupby(["held_out_donor","seed"])["cosine"].median().mean())
    if cosine_low>0 and pearson_low>0 and top>=3 and med>0: relational="CONTEXT_RELATIONAL_GEOMETRY_PRESERVED"
    elif cosine_low<=0 or pearson_low<=0 or med<=0 or top<=1: relational="CONTEXT_RELATIONAL_GEOMETRY_DEGRADED"
    else: relational="CONTEXT_RELATIONAL_GEOMETRY_PARTIAL"
    if amplitude == "CONTEXT_AMPLITUDE_COMPRESSION_STRONG":
        alpha_failure_donors = int((donor_alpha < .5).sum())
    elif amplitude == "CONTEXT_AMPLITUDE_COMPRESSION_MODERATE":
        alpha_failure_donors = int((donor_alpha < .8).sum())
    else:
        alpha_failure_donors = 0
    donor_direction = donor.groupby("held_out_donor")[["operator_cosine", "operator_pearson"]].mean()
    donor_top = donor.groupby("held_out_donor")["top_contrast_correct"].mean()
    donor_fidelity = fidelity.groupby("held_out_donor")["cosine"].median()
    relational_failures = 0
    for held_out_donor in donor_direction.index:
        failed = (
            donor_direction.loc[held_out_donor, "operator_cosine"] <= 0
            or donor_direction.loc[held_out_donor, "operator_pearson"] <= 0
            or donor_top.loc[held_out_donor] <= .5
            or donor_fidelity.loc[held_out_donor] <= 0
        )
        relational_failures += int(failed)
    return amplitude,relational,{"alpha_failure_donors":alpha_failure_donors,"relational_failure_donors":relational_failures}


def run(root: Path, config_path: Path) -> dict[str, Any]:
    config=yaml.safe_load(config_path.read_text(encoding="utf-8")); output=root/config["output_directory"]
    if not (output/"dry_run_audit.json").exists(): raise RuntimeError("Dry-run audit required")
    dry=json.loads((output/"dry_run_audit.json").read_text());
    if not all(dry["leakage_checks"].values()): raise RuntimeError("LODO_BENCHMARK_INVALID")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA expected but unavailable")
    device=torch.device("cuda"); frozen=load_frozen(root); primary=frozen["primary"]
    delta=np.memmap(root/primary["response_path"],mode="r",dtype=np.float32,shape=tuple(primary["response_shape"]))
    ntc=np.load(root/"data/processed/cgc_tcell_1b/ntc_context.float32.npy")
    targets=frozen["targets"]; donors=frozen["donors"]; states=frozen["states"]
    strict=pd.read_csv(root/"results/cgc_tcell/strict_trans_genes.csv"); genes=strict.loc[strict.strict_trans_eligible,"gene_id"].astype(str).tolist(); gene_blocks=np.asarray([hash_block(g) for g in genes])
    anchor_manifest = {
        "definition": "mu_train[p,state] = arithmetic mean delta[p,donor,state] over the three outer-training donors",
        "context_blind_definition": "mu_train[p] = arithmetic mean of mu_train[p,state] over the three states",
        "target_count": len(targets),
        "target_order_sha256": hashlib.sha256("\n".join(targets).encode()).hexdigest(),
        "gene_count": len(genes),
        "gene_order_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        "held_out_response_used": False,
        "folds": [
            {
                "outer_fold": outer,
                "held_out_donor": donors[held],
                "training_donors": [donors[index] for index in range(4) if index != held],
            }
            for outer, held in enumerate(range(4))
        ],
    }
    (output/"intervention_anchor_manifest.json").write_text(
        json.dumps(anchor_manifest, indent=2), encoding="utf-8"
    )
    checkpoint_dir=root/config["checkpoint_directory"]; checkpoint_dir.mkdir(parents=True,exist_ok=True)
    work=root/"data/processed/cgc_tcell_1b"; work.mkdir(parents=True,exist_ok=True)
    writers={name:PredictionWriter(output/f"{name}_predictions.parquet",int(config["response_rank"])) for name in ["mean_anchor","ridge","bilinear","mlp"]}
    metric_rows=[]; distance_rows=[]; fidelity_rows=[]; gene_rows=[]; intervention_rows=[]; inner_rows=[]; seed_rows=[]; checkpoint_rows=[]; basis_rows=[]
    sufficient:dict[str,dict[int,list[dict[str,np.ndarray]]]]={name:{} for name in ["MEAN_STATE_ANCHOR","CONTEXT_BLIND_ANCHOR",*TRAINED_MODELS]}
    peak_vram=0; started=time.perf_counter()
    for outer,held in enumerate(range(4)):
        train=[d for d in range(4) if d!=held]
        fold_predictions: list[tuple[str, int, str, np.ndarray]] = []
        basis_path=checkpoint_dir/f"outer_{outer}_response_basis.npz"; coeff_path=work/f"outer_{outer}_response_coeff.float32.npy"
        components,mean,binfo=fit_response_basis(delta,train,rank=int(config["response_rank"]),oversample=int(config["response_pca_oversample"]),batch_targets=int(config["response_pca_target_batch"]),seed=0,device=device)
        np.savez_compressed(basis_path,components=components,mean=mean,training_donors=np.asarray(train))
        coeff=project_responses(delta,components,coeff_path,donors=train,batch_targets=int(config["response_pca_target_batch"]),device=device)
        if np.isfinite(coeff[:,held]).any():
            raise RuntimeError("LODO_BENCHMARK_INVALID: outer truth loaded before inference")
        h,hmanifest=context_features(ntc,train); context_path=checkpoint_dir/f"outer_{outer}_context.npz"; np.savez_compressed(context_path,**hmanifest)
        basis_rows.append({"outer_fold":outer,"held_out_donor":donors[held],**binfo,"basis_path":str(basis_path.relative_to(root)),"context_path":str(context_path.relative_to(root))})
        for model_name,anchor_kind,residual in [("MEAN_STATE_ANCHOR","MEAN_STATE_ANCHOR",np.zeros((len(targets),3,int(config["response_rank"])),np.float32)),("CONTEXT_BLIND_ANCHOR","CONTEXT_BLIND_ANCHOR",np.zeros((len(targets),3,int(config["response_rank"])),np.float32))]:
            writers["mean_anchor"].append(residual,targets=targets,states=states,outer_fold=outer,held_donor=donors[held],model=model_name,seed=-1,anchor=anchor_kind)
            fold_predictions.append((model_name,-1,anchor_kind,residual))
        alpha,rows=select_ridge_alpha(coeff,h,train,list(map(float,config["ridge_alphas"]))); [row.update({"outer_fold":outer,"held_out_donor":donors[held]}) for row in rows]; inner_rows.extend(rows)
        residual,weights=fit_ridge_predict(coeff,h,train,held,alpha); ridge_path=checkpoint_dir/f"outer_{outer}_ridge.npz"; np.savez_compressed(ridge_path,weights=weights,alpha=alpha,training_donors=np.asarray(train))
        checkpoint_rows.append({"outer_fold":outer,"model":"Ridge","seed":-1,"path":str(ridge_path.relative_to(root)),"sha256":sha256_file(ridge_path),"training_donors":";".join(donors[d] for d in train),"held_out_donor":donors[held]})
        writers["ridge"].append(residual,targets=targets,states=states,outer_fold=outer,held_donor=donors[held],model="Ridge",seed=-1,anchor="MEAN_STATE_ANCHOR")
        fold_predictions.append(("Ridge",-1,"MEAN_STATE_ANCHOR",residual))
        for model_name in ["Bilinear","MLP"]:
            validation=[]
            for valid in train:
                inner_train=[d for d in train if d!=valid]
                for lr in map(float,config["learning_rates"]):
                    for model_seed in map(int,config["seeds"]):
                        model,info=train_neural(model_name,coeff,h,inner_train,valid,learning_rate=lr,seed=model_seed,config=config)
                        row={"outer_fold":outer,"held_out_donor":donors[held],"model":model_name,"inner_validation_donor":donors[valid],"learning_rate":lr,"seed":model_seed,"best_epoch":info["best_epoch"],"validation_response_mse_rank256":info["best_validation_mse"],"epochs_run":info["epochs_run"],"seconds":info["seconds"]}; validation.append(row); inner_rows.append(row)
                        del model; torch.cuda.empty_cache()
            vf=pd.DataFrame(validation); selected_lr=float(vf.groupby("learning_rate").validation_response_mse_rank256.mean().idxmin()); selected_epoch=max(1,int(np.rint(vf.loc[vf.learning_rate.eq(selected_lr),"best_epoch"].median())))
            for model_seed in map(int,config["seeds"]):
                model,info=train_neural(model_name,coeff,h,train,None,learning_rate=selected_lr,seed=model_seed,config=config,fixed_epochs=selected_epoch)
                residual=predict_neural(model,h,held,len(targets),int(config["batch_size"])); model_path=checkpoint_dir/f"outer_{outer}_{model_name.lower()}_seed_{model_seed}.pt"
                torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"training_donors":[donors[d] for d in train],"held_out_donor":donors[held],"learning_rate":selected_lr,"epochs":selected_epoch,"response_basis":str(basis_path.relative_to(root)),"context_features":str(context_path.relative_to(root))},model_path)
                checkpoint_rows.append({"outer_fold":outer,"model":model_name,"seed":model_seed,"path":str(model_path.relative_to(root)),"sha256":sha256_file(model_path),"training_donors":";".join(donors[d] for d in train),"held_out_donor":donors[held]})
                writers[model_name.lower()].append(residual,targets=targets,states=states,outer_fold=outer,held_donor=donors[held],model=model_name,seed=model_seed,anchor="MEAN_STATE_ANCHOR")
                fold_predictions.append((model_name,model_seed,"MEAN_STATE_ANCHOR",residual))
                seed_rows.append({"outer_fold":outer,"held_out_donor":donors[held],"model":model_name,"seed":model_seed,"learning_rate":selected_lr,"epochs":selected_epoch,"training_seconds":info["seconds"],"parameter_count":info["parameter_count"]})
                peak_vram=max(peak_vram,torch.cuda.max_memory_allocated()); del model; torch.cuda.empty_cache()
        # Freeze every prediction before loading any held-out perturbation truth.
        project_response_donors(delta,components,coeff,donors=[held],batch_targets=int(config["response_pca_target_batch"]),device=device)
        evaluator=FoldEvaluator(delta,coeff,components,ntc,train,held,gene_blocks,int(config["gene_chunk"]))
        for model_name,model_seed,anchor_kind,residual in fold_predictions:
            metrics,perp,blocks=evaluator.evaluate(residual,anchor_kind=anchor_kind)
            _collect(metric_rows,distance_rows,fidelity_rows,gene_rows,intervention_rows,metrics,perp,blocks,model_name,model_seed,outer,donors[held],targets)
            sufficient[model_name].setdefault(model_seed,[]).append(perp)
        del evaluator,coeff,components,fold_predictions; torch.cuda.empty_cache()
    for writer in writers.values(): writer.close()
    metrics=pd.DataFrame(metric_rows); distances=pd.DataFrame(distance_rows); fidelity=pd.DataFrame(fidelity_rows); gene=pd.DataFrame(gene_rows); intervention=pd.DataFrame(intervention_rows); inner_frame=pd.DataFrame(inner_rows); seed_frame=pd.DataFrame(seed_rows)
    state=metrics[["model","seed","outer_fold","held_out_donor","state_pearson","state_cosine","state_mse"]].copy(); response=metrics[["model","seed","outer_fold","held_out_donor","response_pearson","response_cosine","response_mse"]].copy(); energy=metrics[["model","seed","outer_fold","held_out_donor","v_true","v_pred","kappa_c"]]; amplitude=metrics[["model","seed","outer_fold","held_out_donor","alpha"]]; direction=metrics[["model","seed","outer_fold","held_out_donor","operator_cosine","operator_pearson"]]
    state.to_csv(output/"state_metrics.csv",index=False); response.to_csv(output/"response_metrics.csv",index=False); energy.to_csv(output/"operator_energy.csv",index=False); amplitude.to_csv(output/"amplitude_calibration.csv",index=False); direction.to_csv(output/"operator_direction.csv",index=False); distances.to_csv(output/"stimulation_distance_retention.csv",index=False); fidelity.to_csv(output/"intervention_operator_fidelity.csv",index=False); gene.to_csv(output/"gene_block_robustness.csv",index=False); intervention.to_csv(output/"intervention_block_robustness.csv",index=False); inner_frame.to_csv(output/"inner_validation_manifest.csv",index=False); seed_frame.to_csv(output/"seed_robustness.csv",index=False); pd.DataFrame(basis_rows).to_csv(output/"response_basis_manifest.csv",index=False)
    contrast=[]
    for (model,seed,outer,donor),frame in distances.groupby(["model","seed","outer_fold","held_out_donor"]):
        truth_order=frame.sort_values("d2_true",ascending=False).contrast.tolist(); pred_order=frame.sort_values("d2_pred",ascending=False).contrast.tolist(); contrast.append({"model":model,"seed":seed,"outer_fold":outer,"held_out_donor":donor,"truth_order":" > ".join(truth_order),"predicted_order":" > ".join(pred_order),"top_contrast_correct":pred_order[0]=="Rest__Stim48hr","top_label":"TOP_CONTRAST_CORRECT" if pred_order[0]=="Rest__Stim48hr" else "TOP_CONTRAST_INCORRECT","full_order_label":"FULL_ORDER_CORRECT" if pred_order==truth_order else "FULL_ORDER_INCORRECT"})
    contrast=pd.DataFrame(contrast); contrast.to_csv(output/"contrast_ordering.csv",index=False)
    donor_summary=metrics.merge(contrast[["model","seed","outer_fold","top_contrast_correct","full_order_label"]],on=["model","seed","outer_fold"])
    donor_summary["kappa_below_one"] = donor_summary["kappa_c"] < 1
    donor_summary["alpha_positive"] = donor_summary["alpha"] > 0
    donor_summary["alpha_below_half"] = donor_summary["alpha"] < .5
    donor_summary["cosine_positive"] = donor_summary["operator_cosine"] > 0
    donor_summary.to_csv(output/"donor_summary.csv",index=False)
    boot_rows=[]
    for mi,model in enumerate(sufficient):
        rows=bootstrap_model(sufficient[model],int(config["bootstrap_draws"]),int(config["bootstrap_seed"])+mi)
        for row in rows: row["model"]=model
        boot_rows.extend(rows)
    bootstrap=pd.DataFrame(boot_rows); bootstrap.to_csv(output/"bootstrap_summary.csv",index=False)
    labels={}; failure={}
    for model in TRAINED_MODELS:
        amp,rel,fail=_classification(model,donor_summary.loc[donor_summary.model.eq(model)],bootstrap.loc[bootstrap.model.eq(model)],fidelity.loc[fidelity.model.eq(model)])
        labels[model]={"amplitude_label":amp,"relational_label":rel}; failure[model]=fail
    failed=[]
    for model in TRAINED_MODELS:
        compressed=labels[model]["amplitude_label"] in {
            "CONTEXT_AMPLITUDE_COMPRESSION_STRONG",
            "CONTEXT_AMPLITUDE_COMPRESSION_MODERATE",
        } or labels[model]["relational_label"]=="CONTEXT_RELATIONAL_GEOMETRY_DEGRADED"
        donor_fail=max(failure[model]["alpha_failure_donors"],failure[model]["relational_failure_donors"])
        if compressed and donor_fail>=3: failed.append(model)
    if len(failed)>=2:
        verdict="CONTEXT_GEOMETRY_COMPRESSION_SUPPORTED"
        amp_count=sum(labels[m]["amplitude_label"] in {"CONTEXT_AMPLITUDE_COMPRESSION_STRONG", "CONTEXT_AMPLITUDE_COMPRESSION_MODERATE"} for m in failed); rel_count=sum(labels[m]["relational_label"]=="CONTEXT_RELATIONAL_GEOMETRY_DEGRADED" for m in failed)
        subtype="AMPLITUDE_AND_RELATIONAL_CGC" if amp_count and rel_count else ("AMPLITUDE_CGC" if amp_count else "RELATIONAL_CGC")
    elif len(failed)==1: verdict="CONTEXT_GEOMETRY_COMPRESSION_MODEL_DEPENDENT"; subtype=None
    elif sum(labels[m]["amplitude_label"]=="CONTEXT_AMPLITUDE_PRESERVED" and labels[m]["relational_label"]=="CONTEXT_RELATIONAL_GEOMETRY_PRESERVED" for m in TRAINED_MODELS)>=2: verdict="CONTEXT_GEOMETRY_COMPRESSION_NOT_SUPPORTED"; subtype=None
    else: verdict="CONTEXT_GEOMETRY_COMPRESSION_INCONCLUSIVE"; subtype=None
    checkpoint_manifest={"checkpoints":checkpoint_rows,"all_state_predictions_same_outer_checkpoint":True}; (output/"model_checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest,indent=2),encoding="utf-8")
    training_manifest={"models":["Ridge","Bilinear","MLP"],"seeds":config["seeds"],"inner_selection_objective":"response_mse_rank256_equivalent","response_pca_rank":config["response_rank"],"posthoc_rescue":False,"transformer":False,"genept":False,"geometry_loss":False,"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda":torch.version.cuda,"peak_vram_bytes":peak_vram}; (output/"model_training_manifest.json").write_text(json.dumps(training_manifest,indent=2),encoding="utf-8")
    verdict_obj={"git_provenance":git(root,"rev-parse","HEAD"),"verdict":verdict,"subtype":subtype,"per_model":labels,"failed_models":failed,"leakage_valid":True}; (output/"verdict.json").write_text(json.dumps(verdict_obj,indent=2),encoding="utf-8")
    run_manifest={"git_provenance":git(root,"rev-parse","HEAD"),"branch":config["branch"],"runtime_seconds":time.perf_counter()-started,"model_fit_count":len(inner_frame.loc[inner_frame.model.isin(["Bilinear","MLP"])])+24+4,"outer_folds":4,"bootstrap_draws":config["bootstrap_draws"],"full_cell_data_used":False,"published_de_targets_used":False,"response_pca_protocol_deviation":True,"peak_vram_bytes":peak_vram,"verdict":verdict}; (output/"run_manifest.json").write_text(json.dumps(run_manifest,indent=2),encoding="utf-8")
    _reports(root,config,verdict_obj,donor_summary,bootstrap,labels)
    return verdict_obj


def _collect(metric_rows,distance_rows,fidelity_rows,gene_rows,intervention_rows,metrics,perp,blocks,model,seed,outer,donor,targets):
    gene_count = 9082
    metric_rows.append({"model":model,"seed":seed,"outer_fold":outer,"held_out_donor":donor,"v_true":metrics["truth2"]/(len(targets)*3*gene_count),"v_pred":metrics["pred2"]/(len(targets)*3*gene_count),"kappa_c":metrics["kappa_c"],"alpha":metrics["alpha"],"operator_cosine":metrics["cosine"],"operator_pearson":metrics["pearson"],**{f"response_{k}":v for k,v in metrics["response_metrics"].items()},**{f"state_{k}":v for k,v in metrics["state_metrics"].items()}})
    for row in metrics["distances"]: distance_rows.append({"model":model,"seed":seed,"outer_fold":outer,"held_out_donor":donor,**row})
    cos=perp["dot"]/np.sqrt(np.maximum(perp["truth2"]*perp["pred2"],1e-30))
    if model == "CONTEXT_BLIND_ANCHOR":
        cos[:] = np.nan
    for p,target in enumerate(targets): fidelity_rows.append({"model":model,"seed":seed,"outer_fold":outer,"held_out_donor":donor,"perturbation_id":target,"cosine":cos[p],"pearson":cos[p]})
    for row in blocks: gene_rows.append({"model":model,"seed":seed,"outer_fold":outer,"held_out_donor":donor,**row})
    for row in intervention_block_rows(perp,targets): intervention_rows.append({"model":model,"seed":seed,"outer_fold":outer,"held_out_donor":donor,**row})


def _markdown_frame(display: pd.DataFrame) -> str:
    columns = display.columns.tolist()
    text = "| " + " | ".join(columns) + " |\n"
    text += "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |\n"
    for row in display.itertuples(index=False, name=None):
        formatted = [str(row[0])]
        for value in row[1:]:
            if isinstance(value, (bool, np.bool_)):
                formatted.append(str(bool(value)))
            elif isinstance(value, (int, np.integer)):
                formatted.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                formatted.append("NA" if np.isnan(value) else f"{float(value):.6f}")
            else:
                formatted.append(str(value))
        text += "| " + " | ".join(formatted) + " |\n"
    return text


def _reports(root,config,verdict,donor,bootstrap,labels):
    reports=root/config["report_directory"]
    output=root/config["output_directory"]
    summary=donor.groupby("model")[["kappa_c","alpha","operator_cosine","operator_pearson","state_pearson","response_pearson","top_contrast_correct"]].mean().reset_index()
    summary["kappa_geometric_mean"] = donor.groupby("model")["kappa_c"].apply(
        lambda values: 0.0 if (values <= 0).any() else float(np.exp(np.log(values).mean()))
    ).reindex(summary["model"]).to_numpy()
    donor_view=donor.groupby(["model","held_out_donor"],as_index=False)[["kappa_c","alpha","operator_cosine","operator_pearson","top_contrast_correct"]].mean()
    consistency=donor_view.groupby("model").agg(
        donors_kappa_below_one=("kappa_c",lambda x:int((x<1).sum())),
        donors_alpha_positive=("alpha",lambda x:int((x>0).sum())),
        donors_alpha_below_half=("alpha",lambda x:int((x<.5).sum())),
        donors_cosine_positive=("operator_cosine",lambda x:int((x>0).sum())),
        donors_top_contrast_correct=("top_contrast_correct",lambda x:int((x>.5).sum())),
    ).reset_index()
    boot_view=bootstrap.loc[bootstrap.metric.isin(["kappa_c","alpha","cosine","pearson",*[f"R_{x}" for x in STATE_PAIR_NAMES]])].copy()
    boot_view=boot_view[["model","metric","point","ci_lower","ci_upper","draws"]]
    fidelity=pd.read_csv(output/"intervention_operator_fidelity.csv")
    fidelity_view=fidelity.groupby("model")["cosine"].agg(
        median="median",q1=lambda x:x.quantile(.25),q3=lambda x:x.quantile(.75),
        positive_fraction=lambda x:(x>0).mean(),fraction_above_0_2=lambda x:(x>.2).mean(),
        fraction_above_0_5=lambda x:(x>.5).mean(),
    ).reset_index()
    basis=pd.read_csv(output/"response_basis_manifest.csv")[["outer_fold","held_out_donor","captured_training_variance_fraction","fit_seconds"]]
    text=_markdown_frame(summary)
    donor_text=_markdown_frame(donor_view)
    consistency_text=_markdown_frame(consistency)
    boot_text=_markdown_frame(boot_view)
    fidelity_text=_markdown_frame(fidelity_view)
    basis_text=_markdown_frame(basis)
    (reports/"cgc_tcell_1b_model_benchmark.md").write_text(f"# CGC-TCELL-1B model benchmark\n\nGit provenance: `{verdict['git_provenance']}` on `{config['branch']}`.\n\nFour strict leave-one-donor-out folds were evaluated. Held-out donor perturbation responses were never used for training, validation, normalization, PCA, feature selection, or checkpoint selection. The held-out donor contributed only its three observed NTC states as test-time context input.\n\n## Model means\n\n{text}\n## Four-donor summary\n\n{donor_text}\n## Donor consistency counts (out of four)\n\n{consistency_text}\n## Frozen 10,000-draw intervention bootstrap\n\n{boot_text}\n## Intervention-wise operator fidelity\n\n{fidelity_text}\n## Training-only response bases\n\n{basis_text}\nPer-model frozen labels:\n\n"+"\n".join(f"- **{m}**: `{v['amplitude_label']}`; `{v['relational_label']}`" for m,v in labels.items())+f"\n\nFormal verdict: **`{verdict['verdict']}`**"+(f" (`{verdict['subtype']}`)" if verdict['subtype'] else "")+".\n",encoding="utf-8")
    (reports/"cgc_tcell_1b_phase_summary.md").write_text(f"# CGC-TCELL-1B phase summary\n\nGit provenance: `{verdict['git_provenance']}` on `{config['branch']}`.\n\nThis is a formal context-generalization benchmark for covered interventions and a completely unseen donor. All four outer folds and three neural seeds completed from one checkpoint per donor/model/seed across all stimulation states. MeanAnchor is descriptive only; the formal decision uses Ridge, bilinear, and MLP.\n\nFinal verdict: **`{verdict['verdict']}`**"+(f"\n\nSubtype: **`{verdict['subtype']}`**" if verdict['subtype'] else "")+"\n\nNo Transformer, GenePT, donor embedding, geometry loss, retrieval model, normalization change, or post-result rescue was used. The result is specific to strict donor generalization in this T-cell stimulation system.\n",encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path.cwd()); parser.add_argument("--config",type=Path,default=Path("configs/cgc_tcell_1b.yaml")); args=parser.parse_args(); root=args.root.resolve(); config=args.config if args.config.is_absolute() else root/args.config; print(json.dumps(run(root,config),indent=2))


if __name__ == "__main__": main()
