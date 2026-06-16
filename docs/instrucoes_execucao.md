# Instrucoes de execucao

## Requisitos

- Python 3.10 ou superior.
- Conta Kaggle apenas se o download automatico pedir autenticacao.
- Bibliotecas listadas em `requirements.txt`.

## Rodar sem deixar o computador lento

Este comando limita o uso de CPU e usa amostra local:

```powershell
python src\train_evaluate.py --sample-size 2500 --n-jobs 1
```

Esse modo gera todos os arquivos de resultado e graficos, mas usa validacao cruzada 3-fold externa e 2-fold interna para reduzir custo computacional.

## Rodar no Google Colab

1. Abra o notebook `notebooks/trabalho_agro_previsao_safra.ipynb` no Google Colab.
2. Envie a pasta do projeto para o Colab ou monte o Google Drive.
3. Execute as celulas em ordem.
4. Para o experimento completo, use:

```bash
python src/train_evaluate.py --full --n-jobs 2
```

## Rodar com GPU no Google Colab

1. No menu do Colab, selecione `Runtime > Change runtime type`.
2. Em `Hardware accelerator`, escolha `T4 GPU`.
3. Execute:

```bash
pip install -r requirements-colab.txt
python src/download_data.py
python src/train_evaluate_gpu_colab.py --folds 5
```

Para testar mais rapido antes da execucao final:

```bash
python src/train_evaluate_gpu_colab.py --folds 3 --sample-size 6000
```

A versao GPU treina:

- XGBoost com `device="cuda"`.
- CatBoost com `task_type="GPU"`.
- MLP em PyTorch usando CUDA.

As saidas ficam em:

- `outputs/results_gpu/model_results_gpu.csv`
- `outputs/results_gpu/resultados_gpu.md`
- `outputs/figures_gpu/gpu_01_comparacao_modelos.png`
- `outputs/figures_gpu/gpu_02_real_vs_predito.png`

## Recriar os resultados

```powershell
python -m pip install -r requirements.txt
python src\download_data.py
python src\train_evaluate.py --sample-size 2500 --n-jobs 1
```

## Saidas geradas

- `data/raw/yield_df.csv`: tabela principal do Kaggle.
- `data/processed/crop_yield_features.csv`: base final com atributos engenheirados.
- `outputs/results/model_results.csv`: RMSE, MAE e R2 de cada modelo.
- `outputs/results/resultados.md`: resumo textual dos resultados.
- `outputs/results/casos_uso.csv`: dois exemplos de entrada e saida dos modelos.
- `outputs/figures/01_distribuicao_rendimento.png`
- `outputs/figures/02_matriz_correlacao.png`
- `outputs/figures/03_rendimento_por_cultura.png`
- `outputs/figures/04_comparacao_modelos.png`
- `outputs/figures/05_real_vs_predito_melhor_modelo.png`
- `outputs/figures/06_importancia_atributos.png`

## Observacao sobre desempenho

O treinamento completo e mais pesado porque combina grid search com validacao cruzada. Para apresentacao, os resultados locais rapidos ja sao funcionais. Para entrega final mais forte, rode o modo completo no Colab.
