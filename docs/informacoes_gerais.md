# Informacoes gerais do trabalho

## Tema

Agronegocio - previsao de safra.

## Tipo de problema

Regressao supervisionada.

## Variavel alvo

`hg/ha_yield`: rendimento agricola em hectogramas por hectare.

## Dataset

Kaggle `patelris/crop-yield-prediction-dataset`.

## Artigo usado para comparacao

Yan et al. (2025), "Crop Yield Time-Series Data Prediction Based on Multiple Hybrid Machine Learning Models".

## Tecnicas de AM da execucao final

- MLP Regressor em PyTorch.
- XGBoost Regressor.
- CatBoost Regressor.

## Metricas

- RMSE: erro quadratico medio com penalizacao maior para erros grandes. Quanto menor, melhor.
- MAE: erro absoluto medio. Quanto menor, melhor.
- R2: proporcao da variacao explicada pelo modelo. Quanto mais perto de 1, melhor.

## Validacao

A execucao final usou validacao cruzada k-fold no Google Colab com 27.644 registros apos engenharia de atributos temporais.

## Resultados finais

| Modelo | RMSE | MAE | R2 |
|---|---:|---:|---:|
| MLP | 9225,74 | 4116,18 | 0,9883 |
| XGBoost | 9362,52 | 3643,92 | 0,9879 |
| CatBoost | 9409,89 | 3777,14 | 0,9878 |

Melhor modelo por RMSE e R2: MLP.

Melhor modelo por MAE: XGBoost.

## Arquivos de resultado

Ao executar o pipeline, os resultados sao gerados em:

- `outputs/results_gpu`
- `outputs/figures_gpu`

Essas pastas nao sao versionadas no Git, pois contem arquivos derivados da execucao.

## Documentos de apoio

- `docs/instrucoes_execucao.md`: como rodar tudo no Google Colab.
- `docs/explicacao_variaveis_treinamento.md`: explicacao das variaveis usadas.
- `docs/explicacao_modelos.md`: explicacao dos modelos treinados.
- `docs/explicacao_figuras.md`: explicacao das figuras geradas.
- `docs/explicacao_codigo_treinamento.md`: explicacao do script de treinamento.
