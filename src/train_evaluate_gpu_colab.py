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


# Caminhos principais do projeto. O script sempre resolve os arquivos a partir da
# raiz do repositorio, entao funciona tanto no Colab quanto em execucao local.
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "yield_df.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "crop_yield_features.csv"
FIG_DIR = ROOT / "outputs" / "figures_gpu"
RES_DIR = ROOT / "outputs" / "results_gpu"

# Variavel que os modelos tentam prever: rendimento agricola em hg/ha.
TARGET = "hg/ha_yield"

# A especificacao pede um conjunto enxuto de atributos. Aqui ficam as 10
# variaveis finais usadas no treinamento.
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

# Apenas variaveis numericas entram na matriz de correlacao. Area e Item sao
# categoricas e entram nos modelos depois de codificacao one-hot.
NUMERIC_FEATURES = [
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
    # Fixa a aleatoriedade para deixar os resultados mais reprodutiveis.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # RMSE e a metrica usada para escolher a melhor configuracao de cada modelo.
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_and_engineer() -> pd.DataFrame:
    # Carrega o CSV consolidado do dataset e cria as variaveis derivadas usadas
    # no experimento.
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {RAW_PATH}. Execute: python src/download_data.py"
        )

    df = pd.read_csv(RAW_PATH)
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])

    # A ordenacao e obrigatoria para criar corretamente atributos historicos.
    # O rendimento anterior deve ser da mesma regiao, mesma cultura e ano anterior.
    df = df.sort_values(["Area", "Item", "Year"]).reset_index(drop=True)

    # Engenharia de atributos climaticos: permite aos modelos aprenderem efeitos
    # nao lineares e combinados de temperatura e chuva.
    df["temp_squared"] = df["avg_temp"] ** 2
    df["rain_temp_interaction"] = (
        df["average_rain_fall_mm_per_year"] * df["avg_temp"]
    )

    # Engenharia de atributos temporais: usa a memoria produtiva de cada par
    # Area-Item, sem misturar culturas ou regioes diferentes.
    grouped = df.groupby(["Area", "Item"], sort=False)[TARGET]
    df["yield_lag_1"] = grouped.shift(1)
    df["yield_rolling_mean_3"] = grouped.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    # As primeiras observacoes de cada serie podem ficar sem historico; por isso
    # sao removidas antes do treino.
    df = df.dropna(subset=FINAL_FEATURES + [TARGET]).reset_index(drop=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[FINAL_FEATURES + [TARGET]].to_csv(PROCESSED_PATH, index=False)
    return df


def prepare_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    # Area e Item sao textos/categorias. O one-hot encoding transforma essas
    # categorias em colunas 0/1 para que os modelos consigam usa-las.
    X = pd.get_dummies(df[FINAL_FEATURES], columns=["Area", "Item"], dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    return X.astype(np.float32), y


def param_grid(grid: dict[str, list]) -> list[dict]:
    # Converte uma grade de parametros em todas as combinacoes a testar.
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def metrics_row(model_name: str, y: np.ndarray, pred: np.ndarray, params: dict) -> dict:
    # Padroniza a linha de resultado salva no CSV final.
    return {
        "modelo": model_name,
        "RMSE": rmse(y, pred),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
        "melhores_parametros": json.dumps(params, ensure_ascii=False),
    }


def cross_validate_xgboost(
    X: pd.DataFrame, y: np.ndarray, folds: int, seed: int, use_cuda: bool
) -> tuple[np.ndarray, dict]:
    # XGBoost e um modelo de boosting: varias arvores sao treinadas em sequencia,
    # cada uma tentando corrigir erros das anteriores.
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

    # Para cada combinacao de hiperparametros, gera predicoes fora da dobra de
    # treino. Assim cada registro e previsto por um modelo que nao viu aquele
    # registro durante o treinamento.
    for params in grid:
        pred = np.zeros_like(y, dtype=np.float32)
        for train_idx, valid_idx in cv.split(X):
            model = XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                device="cuda" if use_cuda else "cpu",
                eval_metric="rmse",
                random_state=seed,
                **params,
            )
            model.fit(X.iloc[train_idx], y[train_idx], verbose=False)
            pred[valid_idx] = model.predict(X.iloc[valid_idx]).astype(np.float32)

        score = rmse(y, pred)
        # A melhor configuracao e escolhida pelo menor RMSE.
        if score < best_score:
            best_score = score
            best_pred = pred
            best_params = params

    return best_pred, best_params


def cross_validate_catboost(
    X: pd.DataFrame, y: np.ndarray, folds: int, seed: int
) -> tuple[np.ndarray, dict]:
    # CatBoost tambem usa boosting com arvores. Aqui ele roda em GPU para reduzir
    # o tempo de treino no Colab.
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

    # O procedimento de validacao e o mesmo do XGBoost: testar parametros,
    # prever cada dobra de validacao e guardar a configuracao com menor RMSE.
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
    # MLP e uma rede neural densa. Ela recebe todas as colunas de entrada e
    # retorna um unico valor numerico: o rendimento previsto.
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_layers:
            # Cada bloco tem camada linear, ReLU para nao linearidade e dropout
            # para reduzir overfitting.
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

    # Treina uma MLP para uma dobra especifica da validacao cruzada.
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

    # Na etapa de avaliacao, o modelo nao atualiza pesos; apenas gera previsoes
    # para a dobra de validacao.
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
    # Grade pequena para manter o treino viavel no Colab, mas ainda comparar
    # arquiteturas de rede neural diferentes.
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
            # Redes neurais treinam melhor quando entradas e alvo estao em
            # escalas parecidas; por isso a MLP usa padronizacao por dobra.
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
            # As predicoes voltam para a unidade original, hg/ha, antes do
            # calculo das metricas.
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
    # Centraliza a geracao dos arquivos finais: metricas, predicoes, resumo em
    # Markdown e figuras usadas nos slides.
    RES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Tabelas principais da execucao.
    results.to_csv(RES_DIR / "model_results_gpu.csv", index=False)
    predictions.to_csv(RES_DIR / "cross_val_predictions_gpu.csv", index=False)
    with (RES_DIR / "model_results_gpu.json").open("w", encoding="utf-8") as f:
        json.dump(results.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    best_name = results.iloc[0]["modelo"]
    best_col = f"pred_{best_name}"

    sns.set_theme(style="whitegrid")

    # Analise exploratoria: distribuicao do alvo que os modelos tentam prever.
    plt.figure(figsize=(9, 5))
    sns.histplot(df[TARGET], bins=40, color="#2f6f6d")
    plt.title("Distribuicao do rendimento agricola")
    plt.xlabel("Rendimento (hg/ha)")
    plt.ylabel("Frequencia")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_01_distribuicao_rendimento.png", dpi=180)
    plt.close()

    # Correlacao apenas entre variaveis numericas. Area e Item nao aparecem aqui
    # porque sao categorias, mas entram no treino via one-hot encoding.
    plt.figure(figsize=(10, 6))
    corr = df[NUMERIC_FEATURES + [TARGET]].corr(numeric_only=True)
    sns.heatmap(corr, cmap="vlag", center=0, annot=False)
    plt.title("Correlacao entre atributos numericos")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_02_matriz_correlacao.png", dpi=180)
    plt.close()

    # Compara culturas pela mediana de rendimento, evitando que poucos valores
    # extremos dominem a leitura visual.
    top_items = df.groupby("Item")[TARGET].median().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_items.values, y=top_items.index, color="#6b8e23")
    plt.title("Mediana do rendimento por cultura")
    plt.xlabel("Rendimento mediano (hg/ha)")
    plt.ylabel("Cultura")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_03_rendimento_por_cultura.png", dpi=180)
    plt.close()

    # Comparacao principal: erros (RMSE/MAE) e poder explicativo (R2).
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    error_df = results.melt(
        id_vars="modelo",
        value_vars=["RMSE", "MAE"],
        var_name="metrica",
        value_name="valor",
    )
    sns.barplot(data=error_df, x="modelo", y="valor", hue="metrica", ax=axes[0])
    axes[0].set_title("Erros dos modelos")
    axes[0].set_xlabel("Modelo")
    axes[0].set_ylabel("Erro em hg/ha")
    axes[0].tick_params(axis="x", rotation=12)

    sns.barplot(data=results, x="modelo", y="R2", color="#4c78a8", ax=axes[1])
    axes[1].set_title("R2 dos modelos")
    axes[1].set_xlabel("Modelo")
    axes[1].set_ylabel("R2")
    axes[1].set_ylim(max(0, results["R2"].min() - 0.02), 1.0)
    axes[1].tick_params(axis="x", rotation=12)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.4f", fontsize=9)
    fig.suptitle("Comparacao dos modelos")
    fig.tight_layout()
    plt.savefig(FIG_DIR / "gpu_04_comparacao_modelos.png", dpi=180)
    # Nome antigo mantido para nao quebrar o roteiro/notebook anterior.
    plt.savefig(FIG_DIR / "gpu_01_comparacao_modelos.png", dpi=180)
    plt.close()

    # Grafico separado para enfatizar as metricas de erro.
    plt.figure(figsize=(8, 5))
    sns.barplot(data=error_df, x="modelo", y="valor", hue="metrica")
    plt.title("RMSE e MAE dos modelos")
    plt.xlabel("Modelo")
    plt.ylabel("Erro em hg/ha")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_05_erros_modelos.png", dpi=180)
    plt.close()

    # Grafico separado para enfatizar o R2.
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results, x="modelo", y="R2", color="#4c78a8")
    plt.title("R2 dos modelos")
    plt.xlabel("Modelo")
    plt.ylabel("R2")
    plt.ylim(max(0, results["R2"].min() - 0.02), 1.0)
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_06_r2_modelos.png", dpi=180)
    plt.close()

    # Real vs predito do melhor modelo. Pontos perto da diagonal indicam boas
    # previsoes.
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=predictions["actual"], y=predictions[best_col], s=14, alpha=0.35)
    max_value = max(predictions["actual"].max(), predictions[best_col].max())
    plt.plot([0, max_value], [0, max_value], color="#b22222", linewidth=1.5)
    plt.title(f"Real vs predito - {best_name}")
    plt.xlabel("Rendimento real (hg/ha)")
    plt.ylabel("Rendimento predito (hg/ha)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_07_real_vs_predito_melhor_modelo.png", dpi=180)
    # Nome antigo mantido para nao quebrar o roteiro/notebook anterior.
    plt.savefig(FIG_DIR / "gpu_02_real_vs_predito.png", dpi=180)
    plt.close()

    # Residuos mostram erro com sinal: positivo = superestimou, negativo =
    # subestimou.
    residuals = predictions[best_col] - predictions["actual"]
    plt.figure(figsize=(9, 5))
    sns.histplot(residuals, bins=50, color="#7f3c8d")
    plt.axvline(0, color="#222222", linewidth=1)
    plt.title(f"Distribuicao dos residuos - {best_name}")
    plt.xlabel("Predito - real (hg/ha)")
    plt.ylabel("Frequencia")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_08_residuos_melhor_modelo.png", dpi=180)
    plt.close()

    # Boxplot dos residuos por modelo para comparar dispersao e outliers.
    residual_rows = []
    for col in [c for c in predictions.columns if c.startswith("pred_")]:
        residual_rows.append(
            pd.DataFrame(
                {
                    "modelo": col.replace("pred_", ""),
                    "residuo": predictions[col] - predictions["actual"],
                }
            )
        )
    residual_df = pd.concat(residual_rows, ignore_index=True)
    if len(residual_df) > 15000:
        residual_df = residual_df.sample(15000, random_state=42)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=residual_df, x="modelo", y="residuo")
    plt.axhline(0, color="#222222", linewidth=1)
    plt.title("Residuos por modelo")
    plt.xlabel("Modelo")
    plt.ylabel("Predito - real (hg/ha)")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_09_residuos_por_modelo.png", dpi=180)
    plt.close()

    # Mesmo grafico real vs predito, mas separado por modelo.
    model_order = results["modelo"].tolist()
    pred_long = []
    for model in model_order:
        col = f"pred_{model}"
        pred_long.append(
            pd.DataFrame(
                {
                    "modelo": model,
                    "real": predictions["actual"],
                    "predito": predictions[col],
                }
            )
        )
    pred_long_df = pd.concat(pred_long, ignore_index=True)
    if len(pred_long_df) > 18000:
        pred_long_df = pred_long_df.sample(18000, random_state=42)
    grid = sns.FacetGrid(pred_long_df, col="modelo", col_wrap=3, height=4, sharex=True, sharey=True)
    grid.map_dataframe(sns.scatterplot, x="real", y="predito", s=10, alpha=0.28)
    max_value = max(predictions["actual"].max(), predictions.filter(like="pred_").max().max())
    for ax in grid.axes.flat:
        ax.plot([0, max_value], [0, max_value], color="#b22222", linewidth=1)
        ax.set_xlabel("Real (hg/ha)")
        ax.set_ylabel("Predito (hg/ha)")
    grid.fig.suptitle("Real vs predito por modelo", y=1.03)
    grid.tight_layout()
    grid.savefig(FIG_DIR / "gpu_10_real_vs_predito_por_modelo.png", dpi=180)
    plt.close()

    # Tendencia temporal media do rendimento no conjunto de dados.
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df.groupby("Year", as_index=False)[TARGET].mean(), x="Year", y=TARGET, marker="o")
    plt.title("Rendimento medio por ano")
    plt.xlabel("Ano")
    plt.ylabel("Rendimento medio (hg/ha)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_11_rendimento_medio_por_ano.png", dpi=180)
    plt.close()

    # Relacao visual entre chuva e rendimento. As cores sao culturas diferentes.
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        data=df.sample(min(8000, len(df)), random_state=42),
        x="average_rain_fall_mm_per_year",
        y=TARGET,
        hue="Item",
        s=14,
        alpha=0.45,
        legend=False,
    )
    plt.title("Chuva anual vs rendimento")
    plt.xlabel("Chuva media anual (mm)")
    plt.ylabel("Rendimento (hg/ha)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_12_chuva_vs_rendimento.png", dpi=180)
    plt.close()

    # Relacao visual entre temperatura e rendimento. A dispersao mostra que a
    # temperatura isolada nao explica todo o rendimento.
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        data=df.sample(min(8000, len(df)), random_state=42),
        x="avg_temp",
        y=TARGET,
        hue="Item",
        s=14,
        alpha=0.45,
        legend=False,
    )
    plt.title("Temperatura media vs rendimento")
    plt.xlabel("Temperatura media")
    plt.ylabel("Rendimento (hg/ha)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_13_temperatura_vs_rendimento.png", dpi=180)
    plt.close()

    # Grafico mantido como apoio, mas menos adequado para slides porque mistura
    # metricas em escalas muito diferentes.
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
    plt.title("Comparacao dos modelos")
    plt.xlabel("Modelo")
    plt.ylabel("Valor")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gpu_14_comparacao_modelos_escala_unica.png", dpi=180)
    plt.close()

    # Resumo textual gerado automaticamente junto com os resultados.
    summary = f"""# Resultados - Google Colab

Dataset: Kaggle `patelris/crop-yield-prediction-dataset`

Registros usados: {len(df):,}

Modelos avaliados: MLP, XGBoost e CatBoost.

## Comparacao dos modelos

{results.to_markdown(index=False)}

Melhor modelo pelo menor RMSE: **{best_name}**.

## Figuras geradas

- `gpu_01_distribuicao_rendimento.png`
- `gpu_02_matriz_correlacao.png`
- `gpu_03_rendimento_por_cultura.png`
- `gpu_04_comparacao_modelos.png`
- `gpu_05_erros_modelos.png`
- `gpu_06_r2_modelos.png`
- `gpu_07_real_vs_predito_melhor_modelo.png`
- `gpu_08_residuos_melhor_modelo.png`
- `gpu_09_residuos_por_modelo.png`
- `gpu_10_real_vs_predito_por_modelo.png`
- `gpu_11_rendimento_medio_por_ano.png`
- `gpu_12_chuva_vs_rendimento.png`
- `gpu_13_temperatura_vs_rendimento.png`

## Comparacao com Estudo X

Yan et al. (2025) reporta Random Forest com R2 = 0,986 e MAE = 348,84; Bagging Regressor com R2 = 0,986 e MAE = 345,51; XGBoost com R2 = 0,973 e MAE = 734,95.

Use esta tabela para atualizar os slides caso o resultado supere ou aproxime o Estudo X.
"""
    (RES_DIR / "resultados_gpu.md").write_text(summary, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    # Parametros de linha de comando para controlar a execucao no Colab.
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Permite rodar mesmo sem CUDA. Use apenas para teste local.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    # A execucao final foi pensada para GPU. O fallback em CPU existe apenas para
    # teste pequeno, porque o treinamento completo pode ficar lento.
    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu_fallback:
        raise RuntimeError(
            "CUDA nao esta disponivel. No Colab, selecione Runtime > Change runtime type > T4 GPU."
        )
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Dispositivo PyTorch: {device}")

    # Carrega e prepara o dataset. Em teste rapido, uma amostra menor reduz o
    # tempo de execucao.
    df = load_and_engineer()
    if args.sample_size and len(df) > args.sample_size:
        df = df.sample(args.sample_size, random_state=args.random_state).reset_index(
            drop=True
        )

    X, y = prepare_matrix(df)
    predictions = pd.DataFrame({"actual": y})
    rows: list[dict] = []

    # Treina e avalia cada modelo. As predicoes sao guardadas para gerar graficos
    # de real vs predito e residuos.
    xgb_pred, xgb_params = cross_validate_xgboost(
        X, y, args.folds, args.random_state, cuda_available
    )
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

    # Ordena pelo menor RMSE; o primeiro da tabela vira o melhor modelo.
    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    save_outputs(df, results, predictions)

    print(results.to_string(index=False))
    print(f"Resultados salvos em: {RES_DIR}")
    print(f"Figuras salvas em: {FIG_DIR}")


if __name__ == "__main__":
    main()


