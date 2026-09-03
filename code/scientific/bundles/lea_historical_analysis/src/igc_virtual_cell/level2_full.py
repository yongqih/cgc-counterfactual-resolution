"""Frozen full-battery estimators for the GSE207049 Level-2 experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit


def pooled_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.square(y_true, dtype=np.float64).sum())
    numerator = float(np.square(y_true - y_pred, dtype=np.float64).sum())
    return 1.0 - numerator / denominator if denominator > 0 else float("nan")


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def standardize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean) / scale).astype(np.float32, copy=False)


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    result = (
        np.square(x64).sum(axis=1, keepdims=True)
        + np.square(y64).sum(axis=1)[None, :]
        - 2 * x64 @ y64.T
    )
    np.maximum(result, 0, out=result)
    return result


def median_squared_distance(x: np.ndarray) -> float:
    distances = squared_distances(x, x)
    values = distances[np.triu_indices(len(x), 1)]
    positive = values[values > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def rbf_kernel_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    gamma_multiplier: float,
    alpha: float,
) -> tuple[np.ndarray, float]:
    mean, scale = standardize_fit(x_train)
    train = standardize_apply(x_train, mean, scale)
    test = standardize_apply(x_test, mean, scale)
    median_distance = median_squared_distance(train)
    gamma = gamma_multiplier / median_distance
    train_kernel = np.exp(-gamma * squared_distances(train, train))
    test_kernel = np.exp(-gamma * squared_distances(test, train))
    y_mean = y_train.mean(axis=0, dtype=np.float64)
    coefficients = np.linalg.solve(
        train_kernel + alpha * np.eye(len(train_kernel)),
        y_train.astype(np.float64) - y_mean,
    )
    prediction = test_kernel @ coefficients + y_mean
    return prediction.astype(np.float32), gamma


def inner_splitter(strata: np.ndarray, seed: int, folds: int = 3):
    if len(np.unique(strata)) > 1:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(
            np.zeros(len(strata)), strata
        )
    return KFold(n_splits=folds, shuffle=True, random_state=seed).split(
        np.zeros(len(strata))
    )


def tune_rbf_kernel(
    x: np.ndarray,
    delta: np.ndarray,
    strata: np.ndarray,
    gamma_multipliers: list[float],
    alphas: list[float],
    seed: int,
) -> tuple[float, float, pd.DataFrame]:
    totals = {(gamma, alpha): [0.0, 0.0] for gamma in gamma_multipliers for alpha in alphas}
    for train, valid in inner_splitter(strata, seed):
        mean_delta = delta[train].mean(axis=0)
        y_train = delta[train] - mean_delta
        y_valid = delta[valid] - mean_delta
        denominator = float(np.square(y_valid, dtype=np.float64).sum())
        for gamma_multiplier, alpha in totals:
            prediction, _ = rbf_kernel_ridge_predict(
                x[train], y_train, x[valid], gamma_multiplier, alpha
            )
            totals[(gamma_multiplier, alpha)][0] += float(
                np.square(y_valid - prediction, dtype=np.float64).sum()
            )
            totals[(gamma_multiplier, alpha)][1] += denominator
    rows = [
        {
            "gamma_multiplier": gamma,
            "alpha": alpha,
            "inner_pooled_R2": 1 - sse / denominator,
        }
        for (gamma, alpha), (sse, denominator) in totals.items()
    ]
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["inner_pooled_R2", "gamma_multiplier", "alpha"],
        ascending=[False, True, True],
    ).iloc[0]
    return float(best["gamma_multiplier"]), float(best["alpha"]), table


@dataclass(frozen=True)
class BoostingSetting:
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    l2_regularization: float


def fit_boosting_program_decoder(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    setting: BoostingSetting,
    baseline_components: int,
    response_components: int,
    seed: int,
) -> np.ndarray:
    mean, scale = standardize_fit(x_train)
    train = standardize_apply(x_train, mean, scale)
    test = standardize_apply(x_test, mean, scale)
    input_k = min(baseline_components, len(train) - 1, train.shape[1])
    input_pca = PCA(n_components=input_k, svd_solver="randomized", random_state=seed)
    z_train = input_pca.fit_transform(train)
    z_test = input_pca.transform(test)
    output_k = min(response_components, len(y_train) - 1, y_train.shape[1])
    output_pca = PCA(n_components=output_k, svd_solver="randomized", random_state=seed + 1)
    target_scores = output_pca.fit_transform(y_train)
    predicted_scores = np.empty((len(test), output_k), dtype=np.float64)
    for component in range(output_k):
        model = HistGradientBoostingRegressor(
            learning_rate=setting.learning_rate,
            max_iter=setting.max_iter,
            max_leaf_nodes=setting.max_leaf_nodes,
            l2_regularization=setting.l2_regularization,
            min_samples_leaf=10,
            early_stopping=False,
            random_state=seed + 10 + component,
        )
        model.fit(z_train, target_scores[:, component])
        predicted_scores[:, component] = model.predict(z_test)
    return output_pca.inverse_transform(predicted_scores).astype(np.float32)


def tune_boosting(
    x: np.ndarray,
    delta: np.ndarray,
    strata: np.ndarray,
    settings: list[BoostingSetting],
    baseline_components: int,
    response_components: int,
    seed: int,
) -> tuple[int, pd.DataFrame]:
    totals = {index: [0.0, 0.0] for index in range(len(settings))}
    for split_number, (train, valid) in enumerate(inner_splitter(strata, seed)):
        mean_delta = delta[train].mean(axis=0)
        y_train = delta[train] - mean_delta
        y_valid = delta[valid] - mean_delta
        denominator = float(np.square(y_valid, dtype=np.float64).sum())
        for index, setting in enumerate(settings):
            prediction = fit_boosting_program_decoder(
                x[train], y_train, x[valid], setting,
                baseline_components, response_components,
                seed + 100 * split_number + index,
            )
            totals[index][0] += float(
                np.square(y_valid - prediction, dtype=np.float64).sum()
            )
            totals[index][1] += denominator
    rows = []
    for index, (sse, denominator) in totals.items():
        rows.append({
            "setting_index": index,
            **settings[index].__dict__,
            "inner_pooled_R2": 1 - sse / denominator,
        })
    table = pd.DataFrame(rows)
    best = int(table.sort_values(
        ["inner_pooled_R2", "setting_index"], ascending=[False, True]
    ).iloc[0]["setting_index"])
    return best, table


class ResidualBlock(torch.nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, width),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DeepResidualMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int = 768, blocks: int = 5) -> None:
        super().__init__()
        self.input = torch.nn.Sequential(
            torch.nn.Linear(input_dim, width), torch.nn.LayerNorm(width), torch.nn.GELU()
        )
        self.blocks = torch.nn.Sequential(*[ResidualBlock(width, 0.2) for _ in range(blocks)])
        self.output = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(x)))


@dataclass(frozen=True)
class DeepSetting:
    learning_rate: float
    weight_decay: float


def fit_input_pca(
    x_train: np.ndarray, x_test: np.ndarray, components: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = standardize_fit(x_train)
    train = standardize_apply(x_train, mean, scale)
    test = standardize_apply(x_test, mean, scale)
    k = min(components, len(train) - 1, train.shape[1])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(train).astype(np.float32), pca.transform(test).astype(np.float32)


def train_deep_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    setting: DeepSetting,
    device: torch.device,
    seed: int,
    max_epochs: int = 150,
    patience_limit: int = 15,
) -> tuple[int, float, pd.DataFrame]:
    torch.manual_seed(seed)
    model = DeepResidualMLP(x_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=32, shuffle=True, generator=generator,
    )
    valid_x = torch.from_numpy(x_valid).to(device)
    valid_y = torch.from_numpy(y_valid).to(device)
    best_loss = float("inf")
    best_epoch = 1
    patience = 0
    rows = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(xb)
            train_n += len(xb)
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(valid_x), valid_y))
        rows.append({"epoch": epoch, "train_MSE": train_loss / train_n, "valid_MSE": valid_loss})
        if valid_loss < best_loss - 1e-7:
            best_loss = valid_loss
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                break
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, best_loss, pd.DataFrame(rows)


def train_deep_fixed(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    setting: DeepSetting,
    epochs: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    model = DeepResidualMLP(x_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=32, shuffle=True, generator=generator,
    )
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x_test), 64):
            predictions.append(model(torch.from_numpy(x_test[start:start + 64]).to(device)).cpu().numpy())
    result = np.concatenate(predictions).astype(np.float32)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def tune_deep(
    x: np.ndarray,
    delta: np.ndarray,
    strata: np.ndarray,
    settings: list[DeepSetting],
    components: int,
    device: torch.device,
    seed: int,
) -> tuple[int, int, pd.DataFrame]:
    if len(np.unique(strata)) > 1:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train, valid = next(splitter.split(x, strata))
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(x))
        valid_n = max(1, int(round(0.2 * len(x))))
        valid, train = order[:valid_n], order[valid_n:]
    mean_delta = delta[train].mean(axis=0)
    y_train = (delta[train] - mean_delta).astype(np.float32)
    y_valid = (delta[valid] - mean_delta).astype(np.float32)
    x_train, x_valid = fit_input_pca(x[train], x[valid], components, seed)
    rows = []
    histories = []
    for index, setting in enumerate(settings):
        epoch, loss, history = train_deep_validation(
            x_train, y_train, x_valid, y_valid, setting, device, seed + index
        )
        history["setting_index"] = index
        histories.append(history)
        rows.append({"setting_index": index, **setting.__dict__, "best_epoch": epoch, "valid_MSE": loss})
    table = pd.DataFrame(rows)
    best = table.sort_values(["valid_MSE", "setting_index"]).iloc[0]
    history_table = pd.concat(histories, ignore_index=True)
    history_table.attrs["selection"] = table.to_dict(orient="records")
    return int(best["setting_index"]), int(best["best_epoch"]), history_table


def fit_deep_outer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    setting: DeepSetting,
    components: int,
    epochs: int,
    seeds: list[int],
    device: torch.device,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    train, test = fit_input_pca(x_train, x_test, components, seeds[0] + fold)
    seed_predictions = []
    for seed in seeds:
        seed_predictions.append(
            train_deep_fixed(train, y_train.astype(np.float32), test, setting, epochs, device, seed + 100 * fold)
        )
    stacked = np.stack(seed_predictions)
    return stacked.mean(axis=0).astype(np.float32), stacked.astype(np.float32)
