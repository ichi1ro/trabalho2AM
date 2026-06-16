# Trabalho 2 - Aprendizado de Maquina no Agronegocio

Tema: previsao de safra no agronegocio.

Dataset: Kaggle `patelris/crop-yield-prediction-dataset`.

Estudo X: Yan et al. (2025), "Crop Yield Time-Series Data Prediction Based on Multiple Hybrid Machine Learning Models".

## O que foi implementado

- Download do dataset do Kaggle.
- Engenharia de atributos para fechar 10 variaveis finais.
- Analise exploratoria com graficos.
- Treinamento de 3 tecnicas: SVM, Random Forest e rede neural MLP.
- Grid Search com validacao cruzada.
- Comparacao com Estudo X.
- Resultados em Markdown, CSV e imagens PNG para slides.

## Execucao rapida local

```powershell
python -m pip install -r requirements.txt
python src\download_data.py
python src\train_evaluate.py --sample-size 2500 --n-jobs 1
```

O modo rapido foi usado para nao travar computadores pessoais. Ele usa uma amostra e uma grade menor.

## Execucao completa

Use preferencialmente no Google Colab:

```bash
pip install -r requirements.txt
python src/download_data.py
python src/train_evaluate.py --full --n-jobs 2
```

## Arquivos principais

- `notebooks/trabalho_agro_previsao_safra.ipynb`: notebook para entregar ou rodar no Colab.
- `src/download_data.py`: baixa/copia os CSVs do Kaggle.
- `src/train_evaluate.py`: prepara dados, treina modelos, gera resultados e graficos.
- `outputs/results/resultados.md`: resumo dos resultados.
- `outputs/results/model_results.csv`: metricas dos modelos.
- `outputs/figures/*.png`: graficos para slides.
- `docs/trabalho_academico.md`: texto base do trabalho.
- `docs/roteiro_slides.md`: conteudo sugerido para slides.
- `docs/instrucoes_execucao.md`: instrucoes detalhadas de execucao.
