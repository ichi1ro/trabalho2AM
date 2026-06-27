# Resultados finais

Execucao final: Google Colab.

Dataset: Kaggle `patelris/crop-yield-prediction-dataset`.

Registros usados apos engenharia de atributos: 27.644.

Modelos avaliados:

- MLP em PyTorch.
- XGBoost.
- CatBoost.

## Metricas finais

| Modelo | RMSE | MAE | R2 |
|---|---:|---:|---:|
| MLP | 9225,74 | 4116,18 | 0,9883 |
| XGBoost | 9362,52 | 3643,92 | 0,9879 |
| CatBoost | 9409,89 | 3777,14 | 0,9878 |

## Interpretacao

O melhor modelo pelo menor RMSE e maior R2 foi a MLP. Isso indica que, no experimento final, a rede neural foi a melhor em termos de erro geral e capacidade de explicar a variacao do rendimento agricola.

O XGBoost teve o menor MAE. Isso significa que, considerando o erro absoluto medio, ele teve previsoes ligeiramente mais estaveis em media.

Os tres modelos ficaram muito proximos, todos com R2 acima de 0,987. Isso indica desempenho alto para a tarefa de regressao.

## Comparacao com Estudo X

O Estudo X, Yan et al. (2025), reporta:

| Modelo no Estudo X | Resultado |
|---|---:|
| Random Forest | R2 = 0,986 |
| Bagging Regressor | R2 = 0,986 |
| XGBoost | R2 = 0,973 |

Comparando com esses valores, a MLP deste trabalho obteve R2 = 0,9883, acima dos resultados reportados no Estudo X. A comparacao deve ser apresentada com cuidado, porque o protocolo de divisao dos dados e validacao pode nao ser identico ao do artigo.

## Figuras principais para slides

As figuras finais estao em `C:\Users\spook\Downloads\resultados_trabalho2\outputs\figures_gpu`.

Use principalmente:

- `gpu_04_comparacao_modelos.png`
- `gpu_05_erros_modelos.png`
- `gpu_06_r2_modelos.png`
- `gpu_07_real_vs_predito_melhor_modelo.png`
- `gpu_09_residuos_por_modelo.png`
