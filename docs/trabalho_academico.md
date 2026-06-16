# Previsao de rendimento agricola com aprendizado de maquina

## 1. Contexto

O tema escolhido foi agronegocio, com foco em previsao de safra. A previsao de rendimento agricola apoia decisoes de planejamento, compra de insumos, gestao de risco e politica publica. O problema foi tratado como regressao: prever `hg/ha_yield`, isto e, rendimento agricola em hectogramas por hectare.

## 2. Dataset escolhido

Foi usado o dataset do Kaggle `patelris/crop-yield-prediction-dataset`.

Link: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset

A tabela principal `yield_df.csv` contem 28.242 registros e as colunas `Area`, `Item`, `Year`, `hg/ha_yield`, `average_rain_fall_mm_per_year`, `pesticides_tonnes` e `avg_temp`. O periodo coberto vai de 1990 a 2013.

## 3. Estudo X

O Estudo X usado para comparacao foi:

Yan et al. (2025), "Crop Yield Time-Series Data Prediction Based on Multiple Hybrid Machine Learning Models".

Link: https://arxiv.org/abs/2502.10405

O estudo usa variaveis equivalentes do dataset de previsao de safra: pais/regiao, cultura, ano, chuva media anual, pesticidas, temperatura media e rendimento. Os resultados reportados incluem Random Forest com R2 = 0,986 e MAE = 348,84; Bagging Regressor com R2 = 0,986 e MAE = 345,51; e XGBoost com R2 = 0,973 e MAE = 734,95.

## 4. Atributos finais

Foram selecionados 10 atributos finais, respeitando a exigencia de 8 a 12 variaveis:

| Atributo | Justificativa |
|---|---|
| `Area` | Representa diferencas geograficas, climaticas e produtivas entre paises/regioes. |
| `Item` | Representa a cultura agricola, que afeta diretamente o rendimento esperado. |
| `Year` | Captura tendencia temporal, evolucao tecnologica e mudancas de manejo. |
| `average_rain_fall_mm_per_year` | Chuva e fator agronomico central para produtividade. |
| `pesticides_tonnes` | Aproxima intensidade de uso de defensivos e manejo agricola. |
| `avg_temp` | Temperatura media influencia desenvolvimento das culturas. |
| `temp_squared` | Captura efeito nao linear da temperatura. |
| `rain_temp_interaction` | Representa interacao entre disponibilidade hidrica e temperatura. |
| `yield_lag_1` | Rendimento do ano anterior por pais e cultura; captura persistencia historica. |
| `yield_rolling_mean_3` | Media movel historica de 3 anos; reduz ruido anual. |

## 5. Pipeline

1. Baixar o dataset do Kaggle.
2. Remover coluna de indice artificial.
3. Ordenar por `Area`, `Item` e `Year`.
4. Criar atributos derivados: temperatura ao quadrado, interacao chuva-temperatura, defasagem de rendimento e media movel.
5. Remover linhas sem historico anterior para as variaveis temporais.
6. Separar `X` e `y`.
7. Aplicar `ColumnTransformer`: imputacao, escalonamento numerico e one-hot encoding de variaveis categoricas.
8. Treinar SVM, Random Forest e MLP com grid search.
9. Avaliar com validacao cruzada.
10. Gerar metricas, graficos e casos de uso.

## 6. Modelos

Foram treinadas tres tecnicas de aprendizado de maquina:

| Modelo | Papel no trabalho |
|---|---|
| SVM LinearSVR | Modelo de margem maxima para regressao. |
| Random Forest Regressor | Ensemble baseado em arvores de decisao. |
| MLP Regressor | Rede neural artificial rasa. |

SVM e MLP usam `TransformedTargetRegressor` para padronizar o alvo durante o treinamento. Isso melhora a estabilidade porque `hg/ha_yield` tem escala numerica alta.

## 7. Resultados obtidos

Resultados do modo rapido/local com 2.500 amostras:

| Modelo | RMSE | MAE | R2 |
|---|---:|---:|---:|
| SVM LinearSVR | 10202,52 | 3108,02 | 0,9861 |
| Random Forest | 10735,80 | 3976,03 | 0,9847 |
| MLP | 11899,04 | 6157,89 | 0,9812 |

O melhor modelo local pelo menor RMSE foi SVM LinearSVR. O resultado em R2 ficou proximo do Estudo X, mas o MAE ficou maior. A comparacao deve ser interpretada com cuidado porque o trabalho local usou amostra e validacao cruzada rapida para evitar travamento do computador.

## 8. Graficos gerados

Os graficos estao em `outputs/figures`:

- `01_distribuicao_rendimento.png`: distribuicao do alvo.
- `02_matriz_correlacao.png`: correlacao entre atributos numericos.
- `03_rendimento_por_cultura.png`: rendimento mediano por cultura.
- `04_comparacao_modelos.png`: metricas por modelo.
- `05_real_vs_predito_melhor_modelo.png`: real versus predito.
- `06_importancia_atributos.png`: importancia por permutacao.

## 9. Limitacoes

- O modo rapido usa amostra para viabilizar execucao local.
- O atributo `yield_lag_1` e muito forte; isso melhora previsao, mas torna o problema mais dependente de historico.
- O dataset e agregado por pais/cultura/ano, sem variaveis locais de solo, manejo ou estagio fenologico.
- A comparacao com o Estudo X nao replica exatamente o mesmo split experimental.

## 10. Conclusao

O trabalho atende aos requisitos do enunciado: dataset do Kaggle, tema de agronegocio, analise descritiva, selecao de 10 variaveis, tres tecnicas de AM, grid search, k-fold cross-validation, comparacao com artigo e geracao de resultados e graficos. Para a entrega final, recomenda-se rodar o modo completo no Google Colab e atualizar os valores da tabela de resultados.

## Referencias

- Kaggle. Crop Yield Prediction Dataset. https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset
- Yan et al. (2025). Crop Yield Time-Series Data Prediction Based on Multiple Hybrid Machine Learning Models. https://arxiv.org/abs/2502.10405
- Scikit-learn documentation. https://scikit-learn.org/stable/
