# Explicacao do codigo de treinamento dos modelos

Arquivo principal: `src/train_evaluate_gpu_colab.py`.

Esse script e responsavel por preparar os dados, treinar os modelos, calcular metricas e gerar os arquivos finais usados no trabalho.

## 1. Objetivo do script

O objetivo do codigo e prever o rendimento agricola (`hg/ha_yield`) a partir de informacoes de pais/regiao, cultura agricola, ano, chuva, pesticidas, temperatura e historico de rendimento.

Como o valor previsto e numerico, o problema e de **regressao supervisionada**.

## 2. Bibliotecas usadas

As principais bibliotecas sao:

| Biblioteca | Papel no codigo |
|---|---|
| `pandas` | Leitura, organizacao e manipulacao da tabela de dados. |
| `numpy` | Operacoes numericas. |
| `scikit-learn` | Metricas, validacao cruzada, padronizacao e apoio ao pipeline. |
| `xgboost` | Treinamento do modelo XGBoost. |
| `catboost` | Treinamento do modelo CatBoost. |
| `torch` | Treinamento da rede neural MLP. |
| `matplotlib` e `seaborn` | Geracao dos graficos para analise e slides. |

## 3. Carregamento e preparacao dos dados

A funcao `load_and_engineer()` carrega o arquivo:

```text
data/raw/yield_df.csv
```

Depois ela remove a coluna artificial `Unnamed: 0`, ordena os dados por `Area`, `Item` e `Year`, e cria atributos derivados.

## 4. Engenharia de atributos

Foram criadas quatro variaveis novas:

| Atributo | O que significa | Por que foi criado |
|---|---|---|
| `temp_squared` | Temperatura media ao quadrado | Captura efeitos nao lineares da temperatura. |
| `rain_temp_interaction` | Chuva multiplicada pela temperatura | Representa a interacao entre agua e temperatura. |
| `yield_lag_1` | Rendimento do ano anterior | Usa historico produtivo como indicio do ano atual. |
| `yield_rolling_mean_3` | Media movel historica de ate 3 anos | Suaviza variacoes anuais e resume tendencia recente. |

Esses atributos ajudam o modelo a capturar padroes agricolas que nao aparecem diretamente nas colunas originais.

## 5. Atributos finais usados

O trabalho usa 10 atributos finais, respeitando a exigencia de 8 a 12 variaveis:

```text
Area
Item
Year
average_rain_fall_mm_per_year
pesticides_tonnes
avg_temp
temp_squared
rain_temp_interaction
yield_lag_1
yield_rolling_mean_3
```

A variavel alvo e:

```text
hg/ha_yield
```

## 6. Transformacao da tabela para treino

A funcao `prepare_matrix()` separa:

- `X`: atributos de entrada.
- `y`: alvo que sera previsto.

As colunas categoricas `Area` e `Item` sao convertidas com `pd.get_dummies()`. Isso transforma textos, como pais e cultura, em colunas numericas que os modelos conseguem processar.

Exemplo conceitual:

```text
Item = Maize
```

vira uma coluna numerica:

```text
Item_Maize = 1
```

## 7. Validacao cruzada k-fold

O codigo usa `KFold` para avaliar os modelos.

A ideia e dividir os dados em partes. Em cada rodada:

1. O modelo treina em algumas partes.
2. O modelo testa na parte restante.
3. O processo se repete ate todas as partes terem sido usadas como teste.

Isso evita depender de uma unica divisao treino/teste e torna a avaliacao mais confiavel.

## 8. Busca de hiperparametros

Cada modelo tem uma pequena grade de parametros.

A funcao `param_grid()` cria todas as combinacoes possiveis desses parametros. O codigo testa cada combinacao e guarda aquela que obteve menor RMSE.

Na pratica, isso cumpre a ideia de **grid search** pedida na especificacao.

## 9. Modelo 1: XGBoost

A funcao `cross_validate_xgboost()` treina o XGBoost.

Resumo do modelo:

> O XGBoost usa varias arvores de decisao em sequencia. Cada nova arvore tenta corrigir os erros das arvores anteriores.

Parametros testados:

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
```

O codigo usa GPU quando disponivel:

```python
device="cuda" if use_cuda else "cpu"
```

## 10. Modelo 2: CatBoost

A funcao `cross_validate_catboost()` treina o CatBoost.

Resumo do modelo:

> O CatBoost tambem e um modelo de boosting com arvores. Ele funciona de forma parecida com o XGBoost na ideia geral e costuma ter bom desempenho em dados tabulares.

Parametros testados:

```text
iterations
depth
learning_rate
l2_leaf_reg
```

No script, ele roda em GPU:

```python
task_type="GPU"
```

## 11. Modelo 3: MLP em PyTorch

A classe `MLPRegressorTorch` define a rede neural.

MLP significa **Multilayer Perceptron**, ou perceptron multicamadas.

Resumo do modelo:

> A MLP e uma rede neural que aprende relacoes nao lineares entre as variaveis de entrada e o rendimento agricola.

A rede e formada por:

1. Camadas lineares (`nn.Linear`).
2. Funcao de ativacao `ReLU`.
3. `Dropout`, para reduzir overfitting.
4. Uma saida numerica final para prever o rendimento.

## 12. Padronizacao na MLP

Na MLP, o codigo usa `StandardScaler` para padronizar tanto os atributos quanto o alvo.

Isso ajuda porque redes neurais treinam melhor quando os valores estao em escalas parecidas.

Depois da predicao, o valor e convertido de volta para a escala original de `hg/ha_yield`.

## 13. Metricas calculadas

A funcao `metrics_row()` calcula:

| Metrica | Interpretacao |
|---|---|
| RMSE | Erro quadratico medio. Penaliza erros grandes. Quanto menor, melhor. |
| MAE | Erro absoluto medio. Quanto menor, melhor. |
| R2 | Quanto da variacao dos dados o modelo explica. Quanto mais perto de 1, melhor. |

## 14. Resultados finais obtidos

Na execucao final do Colab, os resultados foram:

| Modelo | RMSE | MAE | R2 |
|---|---:|---:|---:|
| MLP | 9225,74 | 4116,18 | 0,9883 |
| XGBoost | 9362,52 | 3643,92 | 0,9879 |
| CatBoost | 9409,89 | 3777,14 | 0,9878 |

O melhor modelo por RMSE e R2 foi a **MLP**.

O melhor modelo por MAE foi o **XGBoost**.

## 15. Arquivos gerados

O script salva os resultados em:

```text
outputs/results_gpu/
```

Principais arquivos:

| Arquivo | Conteudo |
|---|---|
| `model_results_gpu.csv` | Tabela com RMSE, MAE, R2 e melhores parametros. |
| `cross_val_predictions_gpu.csv` | Predicoes feitas na validacao cruzada. |
| `resultados_gpu.md` | Resumo textual dos resultados. |

Os graficos sao salvos em:

```text
outputs/figures_gpu/
```

Principais graficos:

- `gpu_04_comparacao_modelos.png`
- `gpu_05_erros_modelos.png`
- `gpu_06_r2_modelos.png`
- `gpu_07_real_vs_predito_melhor_modelo.png`
- `gpu_09_residuos_por_modelo.png`

## 16. Pontos para defesa oral

Pipeline:

> O codigo baixa a base, cria atributos historicos e climaticos, transforma variaveis categoricas em numericas, treina tres modelos de regressao com validacao cruzada e busca de hiperparametros, calcula RMSE, MAE e R2, e gera tabelas e graficos para comparar os modelos.

Melhor modelo:

> A MLP foi melhor pelo RMSE e pelo R2, indicando menor erro geral e maior capacidade de explicar a variacao do rendimento. O XGBoost teve o melhor MAE, indicando menor erro absoluto medio.

Comparacao com o artigo:

> O Estudo X reportou R2 de 0,986 para Random Forest e Bagging. Nosso melhor resultado foi R2 de 0,9883 com MLP. Isso indica desempenho competitivo, mas a comparacao deve considerar possiveis diferencas no protocolo experimental.
