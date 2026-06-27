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
8. Treinar MLP, XGBoost e CatBoost com busca em grade.
9. Avaliar com validacao cruzada.
10. Gerar metricas, graficos e casos de uso.

## 6. Modelos

Foram treinadas tres tecnicas de aprendizado de maquina:

| Modelo | Papel no trabalho |
|---|---|
| MLP Regressor | Rede neural artificial que aprende relacoes nao lineares entre atributos e rendimento. |
| XGBoost Regressor | Modelo de boosting com arvores; cada nova arvore tenta corrigir erros das anteriores. |
| CatBoost Regressor | Modelo de boosting com arvores, semelhante ao XGBoost na ideia geral e forte para dados tabulares. |

Na execucao final do Colab, MLP, XGBoost e CatBoost foram usados porque sao modelos fortes para regressao tabular e conseguem explorar melhor o ambiente acelerado do Colab.

## 7. Resultados obtidos

Resultados finais obtidos no Google Colab com 27.644 registros:

| Modelo | RMSE | MAE | R2 |
|---|---:|---:|---:|
| MLP | 9225,74 | 4116,18 | 0,9883 |
| XGBoost | 9362,52 | 3643,92 | 0,9879 |
| CatBoost | 9409,89 | 3777,14 | 0,9878 |

O melhor modelo pelo menor RMSE foi a MLP. O XGBoost teve o menor MAE, indicando menor erro absoluto medio. Todos os modelos tiveram R2 acima de 0,987, o que indica alta capacidade de explicar a variacao do rendimento agricola no experimento.

Comparando com o Estudo X, o R2 final da MLP foi 0,9883, acima dos valores reportados para Random Forest e Bagging no artigo (0,986) e tambem acima do XGBoost reportado no Estudo X (0,973). A comparacao deve considerar que o protocolo experimental nao e necessariamente identico, mas o resultado e competitivo.

## 8. Graficos gerados

Os graficos finais estao em `outputs/figures_gpu` dentro do pacote `resultados_trabalho2`:

- `gpu_01_distribuicao_rendimento.png`: distribuicao do alvo.
- `gpu_02_matriz_correlacao.png`: correlacao entre atributos numericos.
- `gpu_03_rendimento_por_cultura.png`: rendimento mediano por cultura.
- `gpu_04_comparacao_modelos.png`: comparacao entre MLP, XGBoost e CatBoost.
- `gpu_05_erros_modelos.png`: RMSE e MAE por modelo.
- `gpu_06_r2_modelos.png`: R2 por modelo.
- `gpu_07_real_vs_predito_melhor_modelo.png`: real versus predito do melhor modelo.
- `gpu_09_residuos_por_modelo.png`: residuos por modelo.

## 9. Limitacoes

- O atributo `yield_lag_1` e muito forte; isso melhora previsao, mas torna o problema mais dependente de historico.
- O dataset e agregado por pais/cultura/ano, sem variaveis locais de solo, manejo ou estagio fenologico.
- A comparacao com o Estudo X nao replica exatamente o mesmo split experimental.
- XGBoost e CatBoost sao modelos muito fortes, mas menos intuitivos do que arvores de decisao simples.

## 10. Conclusao

O trabalho atende aos requisitos do enunciado: dataset do Kaggle, tema de agronegocio, analise descritiva, selecao de 10 variaveis, tres tecnicas de AM, busca de hiperparametros, k-fold cross-validation, comparacao com artigo e geracao de resultados e graficos. A MLP obteve o melhor RMSE e o maior R2, enquanto o XGBoost obteve o menor MAE. Os resultados ficaram competitivos em relacao ao Estudo X.

## Referencias

- Kaggle. Crop Yield Prediction Dataset. https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset
- Yan et al. (2025). Crop Yield Time-Series Data Prediction Based on Multiple Hybrid Machine Learning Models. https://arxiv.org/abs/2502.10405
- Scikit-learn documentation. https://scikit-learn.org/stable/

