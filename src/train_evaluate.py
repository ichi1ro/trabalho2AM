from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.compose import TransformedTargetRegressor


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "yield_df.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "crop_yield_features.csv"
FIG_DIR = ROOT / "outputs" / "figures"
RES_DIR = ROOT / "outputs" / "results"

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
CATEGORICAL_FEATURES = ["Area", "Item"]


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


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models(
    random_state: int, fast: bool, n_jobs: int
) -> dict[str, tuple[object, dict[str, list]]]:
    if fast:
        return {
            "SVM_LinearSVR": (
                TransformedTargetRegressor(
                    regressor=Pipeline(
                        [
                            ("prep", make_preprocessor(scale_numeric=True)),
                            (
                                "model",
                                LinearSVR(
                                    random_state=random_state,
                                    max_iter=4000,
                                    dual="auto",
                                ),
                            ),
                        ]
                    ),
                    transformer=StandardScaler(),
                ),
                {
                    "regressor__model__C": [0.1, 1.0],
                    "regressor__model__epsilon": [0.0, 0.1],
                },
            ),
            "RandomForest": (
                Pipeline(
                    [
                        ("prep", make_preprocessor(scale_numeric=False)),
                        (
                            "model",
                            RandomForestRegressor(
                                random_state=random_state,
                                n_jobs=n_jobs,
                            ),
                        ),
                    ]
                ),
                {
                    "model__n_estimators": [120],
                    "model__max_depth": [16, None],
                    "model__min_samples_leaf": [1, 2],
                },
            ),
            "MLP": (
                TransformedTargetRegressor(
                    regressor=Pipeline(
                        [
                            ("prep", make_preprocessor(scale_numeric=True)),
                            (
                                "model",
                                MLPRegressor(
                                    random_state=random_state,
                                    max_iter=250,
                                    early_stopping=True,
                                    validation_fraction=0.15,
                                ),
                            ),
                        ]
                    ),
                    transformer=StandardScaler(),
                ),
                {
                    "regressor__model__hidden_layer_sizes": [(64,), (64, 32)],
                    "regressor__model__alpha": [0.0001],
                    "regressor__model__learning_rate_init": [0.001],
                },
            ),
        }

    return {
        "SVM_LinearSVR": (
            TransformedTargetRegressor(
                regressor=Pipeline(
                    [
                        ("prep", make_preprocessor(scale_numeric=True)),
                        (
                            "model",
                            LinearSVR(
                                random_state=random_state, max_iter=8000, dual="auto"
                            ),
                        ),
                    ]
                ),
                transformer=StandardScaler(),
            ),
            {
                "regressor__model__C": [0.1, 1.0, 10.0],
                "regressor__model__epsilon": [0.0, 0.1, 1.0],
            },
        ),
        "RandomForest": (
            Pipeline(
                [
                    ("prep", make_preprocessor(scale_numeric=False)),
                    (
                        "model",
                        RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
                    ),
                ]
            ),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 18],
                "model__min_samples_leaf": [1, 2],
            },
        ),
        "MLP": (
            TransformedTargetRegressor(
                regressor=Pipeline(
                    [
                        ("prep", make_preprocessor(scale_numeric=True)),
                        (
                            "model",
                            MLPRegressor(
                                random_state=random_state,
                                max_iter=600,
                                early_stopping=True,
                                validation_fraction=0.15,
                            ),
                        ),
                    ]
                ),
                transformer=StandardScaler(),
            ),
            {
                "regressor__model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "regressor__model__alpha": [0.0001, 0.001],
                "regressor__model__learning_rate_init": [0.001],
            },
        ),
    }


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    fast: bool,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    outer_cv = KFold(n_splits=3 if fast else 5, shuffle=True, random_state=random_state)
    inner_cv = KFold(n_splits=2 if fast else 3, shuffle=True, random_state=random_state)
    rows: list[dict] = []
    predictions = pd.DataFrame({"actual": y.to_numpy()})
    best_estimators = {}

    for name, (pipeline, grid) in build_models(random_state, fast, n_jobs).items():
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
        )
        pred = cross_val_predict(search, X, y, cv=outer_cv, n_jobs=1)
        predictions[f"pred_{name}"] = pred
        search.fit(X, y)
        best_estimators[name] = search.best_estimator_

        rows.append(
            {
                "modelo": name,
                "RMSE": rmse(y, pred),
                "MAE": float(mean_absolute_error(y, pred)),
                "R2": float(r2_score(y, pred)),
                "melhores_parametros": json.dumps(
                    search.best_params_, ensure_ascii=False
                ),
            }
        )

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return results, predictions, best_estimators


def save_plots(
    df: pd.DataFrame,
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    best_estimators: dict,
    n_jobs: int,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    plt.figure(figsize=(9, 5))
    sns.histplot(df[TARGET], bins=40, color="#2f6f6d")
    plt.title("Distribuicao do rendimento agricola")
    plt.xlabel("Rendimento (hg/ha)")
    plt.ylabel("Frequencia")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_distribuicao_rendimento.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    corr = df[NUMERIC_FEATURES + [TARGET]].corr(numeric_only=True)
    sns.heatmap(corr, cmap="vlag", center=0, annot=False)
    plt.title("Correlacao entre atributos numericos")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_matriz_correlacao.png", dpi=180)
    plt.close()

    top_items = df.groupby("Item")[TARGET].median().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_items.values, y=top_items.index, color="#6b8e23")
    plt.title("Mediana do rendimento por cultura")
    plt.xlabel("Rendimento mediano (hg/ha)")
    plt.ylabel("Cultura")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_rendimento_por_cultura.png", dpi=180)
    plt.close()

    metric_df = results.melt(
        id_vars="modelo",
        value_vars=["RMSE", "MAE", "R2"],
        var_name="metrica",
        value_name="valor",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=metric_df, x="modelo", y="valor", hue="metrica")
    plt.title("Comparacao dos modelos por validacao cruzada")
    plt.xlabel("Modelo")
    plt.ylabel("Valor")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_comparacao_modelos.png", dpi=180)
    plt.close()

    best_name = results.loc[0, "modelo"]
    pred_col = f"pred_{best_name}"
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=predictions["actual"], y=predictions[pred_col], s=14, alpha=0.35)
    max_value = max(predictions["actual"].max(), predictions[pred_col].max())
    plt.plot([0, max_value], [0, max_value], color="#b22222", linewidth=1.5)
    plt.title(f"Valores reais vs. preditos - {best_name}")
    plt.xlabel("Rendimento real (hg/ha)")
    plt.ylabel("Rendimento predito (hg/ha)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_real_vs_predito_melhor_modelo.png", dpi=180)
    plt.close()

    best_model = best_estimators[best_name]
    sample = df[FINAL_FEATURES].sample(min(1200, len(df)), random_state=42)
    sample_y = df.loc[sample.index, TARGET]
    perm = permutation_importance(
        best_model,
        sample,
        sample_y,
        n_repeats=3,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
    )
    importance = (
        pd.DataFrame({"feature": FINAL_FEATURES, "importance": perm.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(RES_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=importance, x="importance", y="feature", color="#4c78a8")
    plt.title(f"Importancia por permutacao - {best_name}")
    plt.xlabel("Aumento medio do erro ao permutar")
    plt.ylabel("Atributo")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_importancia_atributos.png", dpi=180)
    plt.close()


def write_summary(
    df: pd.DataFrame,
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    fast: bool,
) -> None:
    best = results.iloc[0]
    case_rows = df[FINAL_FEATURES + [TARGET]].sample(2, random_state=7).copy()
    for _, row in results.iterrows():
        case_rows[f"pred_{row['modelo']}"] = predictions.loc[
            case_rows.index, f"pred_{row['modelo']}"
        ].values

    summary = f"""# Resultados do experimento

Dataset: Kaggle `patelris/crop-yield-prediction-dataset`

Registros brutos em `yield_df.csv`: 28.242

Registros usados no experimento: {len(df):,}

Modo de execucao: {"rapido/local" if fast else "completo"}.

Alvo: `{TARGET}` em hectogramas por hectare.

## Atributos finais

{chr(10).join(f'- `{feature}`' for feature in FINAL_FEATURES)}

## Comparacao dos modelos

{results.to_markdown(index=False)}

Melhor modelo pelo menor RMSE: **{best['modelo']}**.

## Comparacao com Estudo X

O Estudo X adotado e Yan et al. (2025), "Crop Yield Time-Series Data Prediction Based on Multiple Hybrid Machine Learning Models". O artigo usa o mesmo conjunto de variaveis do dataset de previsao de safra: `Area`, `Item`, `Year`, `hg/ha_yield`, `average_rain_fall_mm_per_year`, `pesticides_tonnes` e `avg_temp`.

Resultados reportados pelo Estudo X:

- Random Forest: R2 = 0,986; MAE = 348,84; MAPE = 0,103.
- Bagging Regressor: R2 = 0,986; MAE = 345,51; MAPE = 0,101.
- XGBoost: R2 = 0,973; MAE = 734,95; MAPE = 0,209.

Resultado deste trabalho para o melhor modelo:

- {best['modelo']}: R2 = {best['R2']:.4f}; RMSE = {best['RMSE']:.2f}; MAE = {best['MAE']:.2f}.

Observacao metodologica: o modo completo usa validacao cruzada externa com 5 folds e grid search interno com 3 folds. O modo rapido usa 3 folds externos e 2 folds internos para evitar travamentos em computadores pessoais. Diferencas de split, validacao e engenharia de atributos podem explicar diferencas em relacao ao Estudo X.

## Dois casos de uso

{case_rows.to_markdown(index=False)}
"""
    (RES_DIR / "resultados.md").write_text(summary, encoding="utf-8")
    case_rows.to_csv(RES_DIR / "casos_uso.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Executa a grade completa. Use de preferencia no Google Colab.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=6000,
        help="Amostra maxima no modo rapido. Use 0 para usar todos os dados.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Numero de processos paralelos. 1 evita deixar o computador lento.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_engineer()
    fast = not args.full
    if fast and args.sample_size and len(df) > args.sample_size:
        df = df.sample(args.sample_size, random_state=args.random_state).reset_index(
            drop=True
        )

    X = df[FINAL_FEATURES]
    y = df[TARGET]

    results, predictions, best_estimators = evaluate_models(
        X, y, args.random_state, fast=fast, n_jobs=args.n_jobs
    )

    results.to_csv(RES_DIR / "model_results.csv", index=False)
    predictions.to_csv(RES_DIR / "cross_val_predictions.csv", index=False)
    with (RES_DIR / "model_results.json").open("w", encoding="utf-8") as f:
        json.dump(results.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    save_plots(df, results, predictions, best_estimators, n_jobs=args.n_jobs)
    write_summary(df, results, predictions, fast=fast)

    print(results.to_string(index=False))
    print(f"Resultados salvos em: {RES_DIR}")
    print(f"Figuras salvas em: {FIG_DIR}")


if __name__ == "__main__":
    main()
