from __future__ import annotations

from pathlib import Path

import pandas as pd

from train_evaluate_gpu_colab import load_and_engineer, save_outputs


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    results_path = ROOT / "outputs" / "results_gpu" / "model_results_gpu.csv"
    predictions_path = ROOT / "outputs" / "results_gpu" / "cross_val_predictions_gpu.csv"

    if not results_path.exists() or not predictions_path.exists():
        alt_results = ROOT / "resultados_trabalho2" / "outputs" / "results_gpu" / "model_results_gpu.csv"
        alt_predictions = (
            ROOT
            / "resultados_trabalho2"
            / "outputs"
            / "results_gpu"
            / "cross_val_predictions_gpu.csv"
        )
        if alt_results.exists() and alt_predictions.exists():
            results_path = alt_results
            predictions_path = alt_predictions
        else:
            raise FileNotFoundError(
                "Nao encontrei resultados GPU. Rode o treinamento GPU antes."
            )

    df = load_and_engineer()
    predictions = pd.read_csv(predictions_path)
    results = pd.read_csv(results_path)

    if len(df) != len(predictions):
        df = df.iloc[: len(predictions)].copy()

    save_outputs(df, results, predictions)
    print("Figuras GPU regeneradas em outputs/figures_gpu")


if __name__ == "__main__":
    main()
