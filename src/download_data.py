from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub


DATASET = "patelris/crop-yield-prediction-dataset"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(kagglehub.dataset_download(DATASET))

    for csv_path in dataset_path.glob("*.csv"):
        shutil.copy2(csv_path, RAW_DIR / csv_path.name)

    print(f"Dataset Kaggle: {DATASET}")
    print(f"Origem local do cache: {dataset_path}")
    print(f"Arquivos copiados para: {RAW_DIR}")


if __name__ == "__main__":
    main()
