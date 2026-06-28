# Leitura dos resultados em CSV

Este guia descreve os arquivos CSV gerados pelo treinamento e o significado das principais metricas usadas na avaliacao dos modelos.

Os arquivos sao criados em:

```text
outputs/results_gpu
```

## Arquivos principais

| Arquivo | Conteudo |
|---|---|
| `model_results_gpu.csv` | Tabela final de desempenho dos modelos. |
| `cross_val_predictions_gpu.csv` | Predicoes feitas durante a validacao cruzada. |
| `model_results_gpu.json` | Mesmos resultados do CSV final, em formato JSON. |
| `resultados_gpu.md` | Resumo textual gerado pelo script. |

O arquivo mais importante para comparar modelos e `model_results_gpu.csv`.

## `model_results_gpu.csv`

Esse CSV resume o desempenho final dos modelos avaliados.

Colunas:

| Coluna | Significado |
|---|---|
| `modelo` | Nome tecnico do modelo treinado. |
| `RMSE` | Erro quadratico medio com raiz. Quanto menor, melhor. |
| `MAE` | Erro absoluto medio. Quanto menor, melhor. |
| `R2` | Proporcao da variacao do rendimento explicada pelo modelo. Quanto mais perto de 1, melhor. |
| `melhores_parametros` | Configuracao de hiperparametros que teve melhor resultado para aquele modelo. |

Exemplo de leitura:

```text
MLP_PyTorch_CUDA, RMSE = 9225,74, MAE = 4116,18, R2 = 0,9883
```

Interpretacao:

- A MLP foi avaliada na tarefa de prever rendimento agricola.
- O erro RMSE foi de aproximadamente `9225,74 hg/ha`.
- O erro absoluto medio foi de aproximadamente `4116,18 hg/ha`.
- O R2 de `0,9883` indica que o modelo explicou grande parte da variacao observada no rendimento.

## `cross_val_predictions_gpu.csv`

Esse arquivo guarda as predicoes feitas pelos modelos.

Colunas:

| Coluna | Significado |
|---|---|
| `actual` | Valor real do rendimento agricola no dataset. |
| `pred_XGBoost_GPU` | Valor previsto pelo XGBoost. |
| `pred_CatBoost_GPU` | Valor previsto pelo CatBoost. |
| `pred_MLP_PyTorch_CUDA` | Valor previsto pela MLP. |

Cada linha representa um registro avaliado. A diferenca entre a coluna `actual` e uma coluna `pred_...` mostra o erro daquela previsao.

Formula do erro com sinal:

```text
residuo = valor predito - valor real
```

Leitura:

- Residuo perto de `0`: previsao proxima do valor real.
- Residuo positivo: o modelo superestimou o rendimento.
- Residuo negativo: o modelo subestimou o rendimento.

## O que e k-fold

K-fold e uma tecnica de validacao cruzada.

Em vez de separar os dados uma unica vez em treino e teste, o dataset e dividido em `k` partes chamadas folds.

No trabalho, a execucao final usa:

```text
--folds 5
```

Isso significa que os dados sao divididos em 5 partes.

Funcionamento:

1. O modelo treina em 4 partes.
2. O modelo valida na parte que ficou de fora.
3. O processo se repete 5 vezes.
4. No final, todos os registros foram usados uma vez para validacao.

Vantagem:

> A avaliacao fica mais estavel, porque o resultado nao depende de uma unica divisao treino/teste.

No script, isso aparece como:

```python
KFold(n_splits=folds, shuffle=True, random_state=seed)
```

## RMSE

RMSE significa `Root Mean Squared Error`, ou raiz do erro quadratico medio.

Formula conceitual:

```text
RMSE = raiz(media((valor real - valor predito)^2))
```

Como interpretar:

- Mede o tamanho medio do erro.
- Penaliza mais os erros grandes, porque os erros sao elevados ao quadrado.
- Quanto menor, melhor.
- A unidade e a mesma do alvo: `hg/ha`.

No trabalho, o RMSE foi usado para ordenar os modelos e escolher o melhor resultado geral.

## MAE

MAE significa `Mean Absolute Error`, ou erro absoluto medio.

Formula conceitual:

```text
MAE = media(abs(valor real - valor predito))
```

Como interpretar:

- Mede o erro medio sem considerar sinal.
- Nao diferencia se o modelo errou para cima ou para baixo.
- E mais facil de explicar que o RMSE.
- Quanto menor, melhor.
- A unidade tambem e `hg/ha`.

Exemplo:

Se o MAE for `4116,18`, o erro absoluto medio foi de aproximadamente `4116,18 hg/ha`.

## R2

R2 e o coeficiente de determinacao.

Ele indica quanto da variacao do alvo o modelo consegue explicar.

Como interpretar:

| Valor de R2 | Leitura |
|---:|---|
| Proximo de 1 | Modelo explica muito bem a variacao dos dados. |
| Proximo de 0 | Modelo nao melhora muito em relacao a prever uma media. |
| Menor que 0 | Modelo pior que uma previsao simples baseada na media. |

No trabalho, todos os modelos ficaram com R2 acima de `0,987`, indicando desempenho alto.

## Melhores parametros

A coluna `melhores_parametros` mostra a configuracao escolhida para cada modelo.

Exemplo:

```text
{"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03}
```

Esses parametros controlam o comportamento do modelo. O script testa combinacoes e guarda a configuracao com menor RMSE.

Exemplos:

- `n_estimators`: quantidade de arvores no XGBoost.
- `max_depth`: profundidade maxima das arvores.
- `learning_rate`: taxa de aprendizado.
- `hidden_layers`: tamanho das camadas da MLP.
- `dropout`: regularizacao usada na MLP para reduzir overfitting.

## Como comparar os modelos

Regra pratica:

1. Verificar o menor `RMSE`.
2. Verificar o menor `MAE`.
3. Verificar o maior `R2`.
4. Comparar se as diferencas sao grandes ou pequenas.

Resultados finais usados no trabalho:

| Modelo | RMSE | MAE | R2 |
|---|---:|---:|---:|
| MLP | 9225,74 | 4116,18 | 0,9883 |
| XGBoost | 9362,52 | 3643,92 | 0,9879 |
| CatBoost | 9409,89 | 3777,14 | 0,9878 |

Leitura:

- A MLP teve o menor RMSE e o maior R2.
- O XGBoost teve o menor MAE.
- Os tres modelos tiveram resultados proximos.

## Frase para apresentacao

> Os resultados foram avaliados com validacao cruzada k-fold. O RMSE e o MAE medem erro, portanto valores menores indicam melhor desempenho. O R2 mede o quanto o modelo explica da variacao do rendimento, entao valores mais proximos de 1 sao melhores. A MLP teve o melhor RMSE e R2, enquanto o XGBoost teve o menor MAE.
