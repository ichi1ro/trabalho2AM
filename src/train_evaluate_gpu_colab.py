from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "yield_df.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "crop_yield_features.csv"
FIG_DIR = ROOT / "outputs" / "figures_gpu"
RES_DIR = ROOT / "outputs" / "results_gpu"

TARGET = "hg/ha_yield"
FINAL_FEATURES = [
    "Area",
    "Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "temp_squared",
    "rain_temp_interaction",
    "yield_lag_1",
    "yield_rolling_mean_3",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_and_engineer() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {RAW_PATH}. Execute: python src/download_data.py"
        )

    df = pd.read_csv(RAW_PATH)
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    df = df.sort_values(["Area", "Item", "Year"]).reset_index(drop=True)

    df["temp_squared"] = df["avg_temp"] ** 2
    df["rain_temp_interaction"] = (
        df["average_rain_fall_mm_per_year"] * df["avg_temp"]
    )
    grouped = df.groupby(["Area", "Item"], sort=False)[TARGET]
    df["yield_lag_1"] = grouped.shift(1)
    df["yield_rolling_mean_3"] = grouped.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    df = df.dropna(subset=FINAL_FEATURES + [TARGET]).reset_index(drop=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[FINAL_FEATURES + [TARGET]].to_csv(PROCESSED_PATH, index=False)
    return df


def prepare_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.get_dummies(df[FINAL_FEATURES], columns=["Area", "Item"], dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    return X.astype(np.float32), y


def param_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def metrics_row(model_name: str, y: np.ndarray, pred: np.ndarray, params: dict) -> dict:
    return {
        "modelo": model_name,
        "RMSE": rmse(y, pred),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
        "melhores_parametros": json.dumps(params, ensure_ascii=False),
    }


def cross_validate_xgboost(
    X: pd.DataFrame, y: np.ndarray, folds: int, seed: int
) -> tuple[np.ndarray, dict]:
    grid = param_grid(
        {
            "n_estimators": [500, 800],
            "max_depth": [4, 6],
            "learning_rate": [0.03, 0.06],
            "subsample": [0.85],
            "colsample_bytree": [0.85],
        }
    )
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    best_score = float("inf")
    best_pred = np.zeros_like(y, dtype=np.float32)
    best_params: dict = {}

    for params in grid:
        pred = np.zeros_like(y, dtype=np.float32)
        for train_idx, valid_idx in cv.split(X):
            model = XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                device="cuda",
                eval_metric="rmse",
                random_state=seed,
                **params,
            )
            model.fit(X.iloc[train_idx], y[train_idx], verbose=False)
            pred[valid_idx] = model.predict(X.iloc[valid_idx]).astype(np.float32)

        score = rmse(y, pred)
        if score < best_score:
            best_score = score
            best_pred = pred
            best_params = params

    return best_pred, best_params


def cross_validate_catboost(
    X: pd.DataFrame, y: np.ndarray, folds: int, seed: int
) -> tuple[np.ndarray, dict]:
    grid = param_grid(
        {
            "iterations": [600, 900],
            "depth": [6, 8],
            "learning_rate": [0.03, 0.06],
            "l2_leaf_reg": [3.0],
        }
    )
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    best_score = float("inf")
    best_pred = np.zeros_like(y, dtype=np.float32)
    best_params: dict = {}

    for params in grid:
        pred = np.zeros_like(y, dtype=np.float32)
        for train_idx, valid_idx in cv.split(X):
            model = CatBoostRegressor(
                loss_function="RMSE",
                task_type="GPU",
                devices="0",
                random_seed=seed,
                verbose=False,
                **params,
            )
            model.fit(X.iloc[train_idx], y[train_idx])
            pred[valid_idx] = model.predict(X.iloc[valid_idx]).astype(np.float32)

        score = rmse(y, pred)
        if score < best_score:
            best_score = score
            best_pred = pred
            best_params = params

    return best_pred, best_params


class MLPRegressorTorch(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_layers:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_torch_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    params: dict,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    set_seed(seed)
    model = MLPRegressorTorch(
        input_dim=X_train.shape[1],
        hidden_layers=params["hidden_layers"],
        dropout=params["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    model.train()
    for _ in range(params["epochs"]):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    preds: list[np.ndarray] = []
    valid_tensor = torch.from_numpy(X_valid).float()
    valid_loader = DataLoader(valid_tensor, batch_size=4096, shuffle=False)
    with torch.no_grad():
        for xb in valid_loader:
            preds.append(model(xb.to(device)).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def cross_validate_torch_mlp(
    X: pd.DataFrame, y: np.ndarray, folds: int, seed: int, device: torch.device
) -> tuple[np.ndarray, dict]:
    grid = param_grid(
        {
            "hidden_layers": [(128, 64), (256, 128)],
            "dropout": [0.05],
            "lr": [0.001],
            "weight_decay": [0.0001],
            "batch_size": [1024],
            "epochs": [120],
        }
    )
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    best_score = float("inf")
    best_pred = np.zeros_like(y, dtype=np.float32)
    best_params: dict = {}

    X_np = X.to_numpy(dtype=np.float32)
    for params in grid:
        pred_scaled = np.zeros_like(y, dtype=np.float32)
        for train_idx, valid_idx in cv.split(X_np):
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            X_train = x_scaler.fit_transform(X_np[train_idx]).astype(np.float32)
            X_valid = x_scaler.transform(X_np[valid_idx]).astype(np.float32)
            y_train = (
                y_scaler.fit_transform(y[train_idx].reshape(-1, 1))
                .ravel()
                .astype(np.float32)
            )
            fold_pred_scaled = train_torch_fold(
                X_train, y_train, X_valid, params, seed, device
            )
            pred_scaled[valid_idx] = y_scaler.inverse_transform(
                fold_pred_scaled.reshape(-1, 1)
            ).ravel()

        score = rmse(y, pred_scaled)
        if score < best_score:
            best_score = score
            best_pred = pred_scaled
            best_params = params

    json_ready_params = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in best_params.items()
    }
    return best_pred, json_ready_params


def save_outputs(df: pd.DataFrame, results: pd.DataFrame, predictions: pd.DataFrame) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(RES_DIR / "model_results_gpu.csv", index=False)
    predictions.to_csv(RES_DIR / "cross_val_predictions_gpu.csv", index=False)
    with (RES_DIR / "model_results_gpu.json").open("w", encoding="utf-8") as f:
        json.dump(results.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    best_name = results.iloc[0]["modelo"]
    best_col = f"pred_{best_name}"

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=results.melt(
            id_vars="modelo",
            value_vars=["RMSE", "MAE", "R2"],
            var_name="metrica",
            value_name="valor",
        ),
        x="modelo",
        y="valor",
        hue="metrica",
    )
    plt.title("Comparacao dos modelos GPU")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_01_comparacao_modelos.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=predictions["actual"], y=predictions[best_col], s=14, alpha=0.35)
    max_value = max(predictions["actual"].max(), predictions[best_col].max())
    plt.plot([0, max_value], [0, max_value], color="#b22222", linewidth=1.5)
    plt.title(f"Real vs predito - {best_name}")
    plt.xlabel("Rendimento real (hg/ha)")
    plt.ylabel("Rendimento predito (hg/ha)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_02_real_vs_predito.png", dpi=180)
    plt.close()

    summary = f"""# Resultados GPU - Google Colab

Dataset: Kaggle `patelris/crop-yield-prediction-dataset`

Registros usados: {len(df):,}

Modelos GPU: XGBoost GPU, CatBoost GPU e MLP PyTorch CUDA.

## Comparacao dos modelos

{results.to_markdown(index=False)}

Melhor modelo pelo menor RMSE: **{best_name}**.

## Comparacao com Estudo X

Yan et al. (2025) reporta Random Forest com R2 = 0,986 e MAE = 348,84; Bagging Regressor com R2 = 0,986 e MAE = 345,51; XGBoost com R2 = 0,973 e MAE = 734,95.

Use esta tabela para atualizar os slides caso o resultado GPU supere ou aproxime o Estudo X.
"""
    (RES_DIR / "resultados_gpu.md").write_text(summary, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Permite rodar mesmo sem CUDA. Nao recomendado para a entrega GPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu_fallback:
        raise RuntimeError(
            "CUDA nao esta disponivel. No Colab, selecione Runtime > Change runtime type > T4 GPU."
        )
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Dispositivo PyTorch: {device}")

    df = load_and_engineer()
    if args.sample_size and len(df) > args.sample_size:
        df = df.sample(args.sample_size, random_state=args.random_state).reset_index(
            drop=True
        )

    X, y = prepare_matrix(df)
    predictions = pd.DataFrame({"actual": y})
    rows: list[dict] = []

    xgb_pred, xgb_params = cross_validate_xgboost(X, y, args.folds, args.random_state)
    predictions["pred_XGBoost_GPU"] = xgb_pred
    rows.append(metrics_row("XGBoost_GPU", y, xgb_pred, xgb_params))

    cat_pred, cat_params = cross_validate_catboost(X, y, args.folds, args.random_state)
    predictions["pred_CatBoost_GPU"] = cat_pred
    rows.append(metrics_row("CatBoost_GPU", y, cat_pred, cat_params))

    mlp_pred, mlp_params = cross_validate_torch_mlp(
        X, y, args.folds, args.random_state, device
    )
    predictions["pred_MLP_PyTorch_CUDA"] = mlp_pred
    rows.append(metrics_row("MLP_PyTorch_CUDA", y, mlp_pred, mlp_params))

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    save_outputs(df, results, predictions)

    print(results.to_string(index=False))
    print(f"Resultados GPU salvos em: {RES_DIR}")
    print(f"Figuras GPU salvas em: {FIG_DIR}")


if __name__ == "__main__":
    main()
