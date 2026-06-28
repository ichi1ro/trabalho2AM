# Trabalho 2 - Aprendizado de Maquina no Agronegocio

Tema: previsao de safra no agronegocio.

Dataset: Kaggle `patelris/crop-yield-prediction-dataset`.

Estudo X: Yan et al. (2025), "Crop Yield Time-Series Data Prediction Based on Multiple Hybrid Machine Learning Models".

## O que foi implementado

- Download do dataset do Kaggle.
- Engenharia de atributos para fechar 10 variaveis finais.
- Analise exploratoria com graficos.
- Treinamento de 3 tecnicas: MLP, XGBoost e CatBoost.
- Busca de hiperparametros com validacao cruzada.
- Comparacao com Estudo X.
- Resultados em Markdown, CSV e imagens PNG para slides.

## Resultados finais

Ao executar o pipeline, os arquivos de saida sao gravados em `outputs/results_gpu` e `outputs/figures_gpu`.

| Modelo | RMSE | MAE | R2 |
|---|---:|---:|---:|
| MLP | 9225,74 | 4116,18 | 0,9883 |
| XGBoost | 9362,52 | 3643,92 | 0,9879 |
| CatBoost | 9409,89 | 3777,14 | 0,9878 |

Melhor modelo por RMSE e R2: MLP.

Melhor modelo por MAE: XGBoost.

## Arquivos principais

- `notebooks/trabalho_agro_previsao_safra.ipynb`: notebook principal para entrega e execucao no Colab. A primeira celula prepara o acesso ao repositorio.
- `src/download_data.py`: baixa/copia os CSVs do Kaggle.
- `src/train_evaluate_gpu_colab.py`: versao usada no Colab.
- `docs/trabalho_academico.md`: texto base do trabalho.
- `docs/roteiro_slides.md`: conteudo sugerido para slides.
- `docs/instrucoes_execucao.md`: instrucoes detalhadas de execucao.
- `docs/informacoes_gerais.md`: resumo geral do trabalho.
- `docs/explicacao_codigo_treinamento.md`: explicacao didatica do codigo de treinamento.
- `docs/explicacao_variaveis_treinamento.md`: explicacao de cada variavel usada no treinamento.
- `docs/explicacao_modelos.md`: explicacao aprofundada dos modelos MLP, XGBoost e CatBoost.
- `docs/explicacao_figuras.md`: explicacao de cada figura gerada.

## Figuras finais

As figuras geradas ficam em `outputs/figures_gpu`. Para a apresentacao, as mais relevantes sao:

- `gpu_04_comparacao_modelos.png`
- `gpu_05_erros_modelos.png`
- `gpu_06_r2_modelos.png`
- `gpu_07_real_vs_predito_melhor_modelo.png`
- `gpu_09_residuos_por_modelo.png`
