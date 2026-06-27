# Instrucoes de execucao

## Requisitos

- Python 3.10 ou superior.
- Conta Kaggle apenas se o download automatico pedir autenticacao.
- Bibliotecas listadas em `requirements-colab.txt` para execucao no Colab.

## Execucao final usada no trabalho

Os resultados finais usados no trabalho ja estao em:

`C:\Users\spook\Downloads\resultados_trabalho2\outputs`

Essa execucao treinou:

- MLP em PyTorch.
- XGBoost.
- CatBoost.

## Rodar novamente no Google Colab

1. Abra o notebook `notebooks/trabalho_agro_previsao_safra.ipynb` no Google Colab.
2. No menu do Colab, selecione `Runtime > Change runtime type`.
3. Em `Hardware accelerator`, escolha `T4 GPU`.
4. Execute:

```bash
pip install -r requirements-colab.txt
python src/download_data.py
python src/train_evaluate_gpu_colab.py --folds 5
```

Para testar mais rapido antes da execucao final:

```bash
python src/train_evaluate_gpu_colab.py --folds 3 --sample-size 6000
```

## Saidas finais

- `outputs/results_gpu/model_results_gpu.csv`
- `outputs/results_gpu/resultados_gpu.md`
- `outputs/figures_gpu/gpu_01_distribuicao_rendimento.png`
- `outputs/figures_gpu/gpu_02_matriz_correlacao.png`
- `outputs/figures_gpu/gpu_03_rendimento_por_cultura.png`
- `outputs/figures_gpu/gpu_04_comparacao_modelos.png`
- `outputs/figures_gpu/gpu_05_erros_modelos.png`
- `outputs/figures_gpu/gpu_06_r2_modelos.png`
- `outputs/figures_gpu/gpu_07_real_vs_predito_melhor_modelo.png`
- `outputs/figures_gpu/gpu_08_residuos_melhor_modelo.png`
- `outputs/figures_gpu/gpu_09_residuos_por_modelo.png`
- `outputs/figures_gpu/gpu_10_real_vs_predito_por_modelo.png`
- `outputs/figures_gpu/gpu_11_rendimento_medio_por_ano.png`
- `outputs/figures_gpu/gpu_12_chuva_vs_rendimento.png`
- `outputs/figures_gpu/gpu_13_temperatura_vs_rendimento.png`

## Resultados finais

| Modelo | RMSE | MAE | R2 |
|---|---:|---:|---:|
| MLP | 9225,74 | 4116,18 | 0,9883 |
| XGBoost | 9362,52 | 3643,92 | 0,9879 |
| CatBoost | 9409,89 | 3777,14 | 0,9878 |

## Observacao sobre nomes

Nos arquivos gerados pelo Colab, os modelos aparecem como `MLP_PyTorch_CUDA`, `XGBoost_GPU` e `CatBoost_GPU`. Nos slides, use os nomes simplificados:

- `MLP_PyTorch_CUDA` -> MLP
- `XGBoost_GPU` -> XGBoost
- `CatBoost_GPU` -> CatBoost
