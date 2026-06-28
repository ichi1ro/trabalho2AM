# Guia das figuras geradas

As figuras do pipeline sao geradas em:

`outputs/figures_gpu`

O conjunto cobre tres partes da analise: exploracao dos dados, comparacao dos modelos e avaliacao dos erros.

## Convencoes de nome

- O prefixo `gpu_` indica apenas que as figuras vieram da execucao feita no Google Colab com GPU.
- Na apresentacao, os modelos podem ser chamados apenas de MLP, XGBoost e CatBoost.
- Nos arquivos de resultados, os nomes aparecem como `MLP_PyTorch_CUDA`, `XGBoost_GPU` e `CatBoost_GPU`.

## Resultados de referencia

| Modelo | RMSE | MAE | R2 |
|---|---:|---:|---:|
| MLP | 9225,74 | 4116,18 | 0,9883 |
| XGBoost | 9362,52 | 3643,92 | 0,9879 |
| CatBoost | 9409,89 | 3777,14 | 0,9878 |

RMSE e MAE devem ser minimizados. O R2 deve ficar o mais proximo possivel de 1. Nesta execucao, a MLP teve o melhor RMSE e o melhor R2; o XGBoost teve o menor MAE.

## `gpu_01_distribuicao_rendimento.png`

Mostra a distribuicao do alvo `hg/ha_yield`, ou seja, o rendimento agricola.

Interpretação:

- Existem valores de rendimento bem diferentes entre culturas e regioes.
- A distribuicao nao e perfeitamente uniforme.
- Isso mostra que o problema e mais complexo do que prever uma media simples.

Uso nos slides:

> Este grafico mostra que o rendimento agricola varia bastante. Por isso faz sentido usar modelos de aprendizado de maquina, pois eles conseguem capturar padroes entre cultura, regiao, clima e historico produtivo.

## `gpu_02_matriz_correlacao.png`

Mostra a correlacao entre variaveis numericas.

Interpretação:

- Variaveis historicas de rendimento tendem a ter relacao forte com o alvo.
- Chuva, temperatura e pesticidas tambem ajudam, mas isoladamente podem nao explicar tudo.
- A matriz ajuda a entender relacoes lineares entre atributos.

Uso nos slides:

> A matriz de correlacao ajuda a identificar quais variaveis se relacionam mais com o rendimento. As variaveis historicas aparecem como muito importantes, o que faz sentido porque regioes e culturas tendem a manter padroes produtivos ao longo do tempo.

## `gpu_03_rendimento_por_cultura.png`

Mostra a mediana do rendimento para cada cultura agricola.

Interpretação:

- Culturas diferentes possuem escalas de produtividade diferentes.
- Comparar culturas sem considerar o tipo de cultura seria injusto.
- Por isso `Item` foi mantido como atributo do modelo.

Uso nos slides:

> Este grafico mostra que cada cultura tem um nivel produtivo proprio. Por isso o modelo precisa saber qual cultura esta sendo analisada, ja que milho, arroz, batata e soja nao seguem exatamente o mesmo padrao de rendimento.

## `gpu_04_comparacao_modelos.png`

Compara os modelos em dois blocos:

- RMSE e MAE, onde menor e melhor.
- R2, onde maior e melhor.

Interpretação:

- A MLP teve o menor RMSE e o maior R2.
- O XGBoost teve o menor MAE.
- Os tres modelos ficaram muito proximos.

Uso nos slides:

> Este grafico resume o desempenho dos modelos. A MLP foi melhor em RMSE e R2, enquanto o XGBoost teve o menor MAE. Como os resultados ficaram proximos, isso indica que os tres modelos aprenderam bem os padroes do dataset.

## `gpu_01_comparacao_modelos.png`

E uma copia/versao equivalente de `gpu_04_comparacao_modelos.png`.

Uso recomendado:

- Use apenas uma das duas nos slides.
- Preferir `gpu_04_comparacao_modelos.png`, pois ela segue melhor a numeracao do roteiro final.

## `gpu_05_erros_modelos.png`

Mostra RMSE e MAE de cada modelo.

Interpretação:

- RMSE penaliza mais erros grandes.
- MAE mostra o erro absoluto medio.
- A MLP tem melhor RMSE.
- O XGBoost tem melhor MAE.

Uso nos slides:

> Aqui comparamos os erros dos modelos. O RMSE mostra o impacto de erros maiores, enquanto o MAE mostra o erro medio absoluto. A MLP teve o melhor RMSE, mas o XGBoost teve o melhor MAE.

## `gpu_06_r2_modelos.png`

Mostra o R2 dos modelos.

Interpretação:

- Todos os modelos ficaram acima de 0,987.
- A MLP teve o maior R2: 0,9883.
- Isso indica alta capacidade de explicar a variacao do rendimento.

Uso nos slides:

> O R2 mede quanto da variacao do rendimento o modelo consegue explicar. Como todos ficaram perto de 1, os modelos tiveram desempenho alto. A MLP teve o melhor resultado geral por essa metrica.

## `gpu_07_real_vs_predito_melhor_modelo.png`

Mostra valores reais contra valores preditos para o melhor modelo.

Interpretação:

- Cada ponto representa uma observacao.
- Quanto mais perto da linha diagonal, melhor a previsao.
- Pontos longe da diagonal representam erros maiores.

Uso nos slides:

> Este grafico compara o valor real com o valor previsto. Quando os pontos ficam proximos da diagonal, significa que o modelo acertou bem. A concentracao perto da linha indica boa qualidade de previsao.

## `gpu_02_real_vs_predito.png`

E uma copia/versao equivalente de `gpu_07_real_vs_predito_melhor_modelo.png`.

Uso recomendado:

- Use apenas uma das duas nos slides.
- Preferir `gpu_07_real_vs_predito_melhor_modelo.png`.

## `gpu_08_residuos_melhor_modelo.png`

Mostra a distribuicao dos residuos do melhor modelo.

Residuo significa:

```text
predicao - valor real
```

Interpretação:

- Residuos perto de zero indicam previsoes boas.
- Residuos muito positivos indicam superestimativa.
- Residuos muito negativos indicam subestimativa.

Uso nos slides:

> O grafico de residuos mostra onde o modelo erra. Quanto mais concentrados os erros estiverem perto de zero, melhor. Isso ajuda a avaliar se o modelo esta errando de forma muito espalhada ou controlada.

## `gpu_09_residuos_por_modelo.png`

Compara os residuos dos tres modelos.

Interpretação:

- Ajuda a ver qual modelo teve erros mais concentrados.
- Modelos com caixas menores tendem a ter erros mais estaveis.
- Pontos extremos indicam casos em que o modelo errou mais.

Uso nos slides:

> Este grafico compara os erros dos modelos. Ele ajuda a enxergar nao so a media das metricas, mas tambem a dispersao dos erros. Modelos com residuos mais concentrados tendem a ser mais estaveis.

## `gpu_10_real_vs_predito_por_modelo.png`

Mostra real versus predito para cada modelo separadamente.

Interpretação:

- Permite comparar visualmente MLP, XGBoost e CatBoost.
- Se os tres graficos ficam parecidos, os modelos tiveram comportamento semelhante.
- Pontos proximos da diagonal indicam boas previsoes.

Uso nos slides:

> Aqui vemos a comparacao real versus predito para cada modelo. Os tres modelos seguem uma tendencia parecida, o que confirma que todos capturaram bem os principais padroes dos dados.

## `gpu_11_rendimento_medio_por_ano.png`

Mostra o rendimento medio ao longo dos anos.

Interpretação:

- Ajuda a perceber tendencias temporais.
- Pode indicar aumento, queda ou oscilacao da produtividade media.
- Justifica o uso da variavel `Year` e de atributos historicos.

Uso nos slides:

> Este grafico mostra como o rendimento medio muda ao longo do tempo. Isso reforca que o ano e o historico produtivo sao informacoes importantes para prever o rendimento.

## `gpu_12_chuva_vs_rendimento.png`

Mostra a relacao entre chuva anual e rendimento.

Interpretação:

- A relacao nao e perfeitamente linear.
- Muita ou pouca chuva pode afetar culturas de formas diferentes.
- A chuva sozinha nao explica todo o rendimento.

Uso nos slides:

> Este grafico mostra que a chuva influencia o rendimento, mas nao explica tudo sozinha. Por isso o modelo combina chuva com outras variaveis, como cultura, temperatura e historico.

## `gpu_13_temperatura_vs_rendimento.png`

Mostra a relacao entre temperatura media e rendimento.

Interpretação:

- A relacao entre temperatura e rendimento tambem nao e simples.
- Diferentes culturas podem responder de formas diferentes a mesma temperatura.
- Isso justifica o uso de `avg_temp` e `temp_squared`.

Uso nos slides:

> A temperatura afeta o desenvolvimento das culturas, mas seu efeito nao e necessariamente linear. Por isso criamos tambem a temperatura ao quadrado, para ajudar os modelos a capturar efeitos mais complexos.

## `gpu_14_comparacao_modelos_escala_unica.png`

Mostra RMSE, MAE e R2 no mesmo eixo.

Interpretação:

- Como RMSE e MAE sao valores grandes e R2 fica perto de 1, o R2 praticamente some.
- Esse grafico e menos adequado para apresentacao.
- Por isso foi criada a versao em dois paineis (`gpu_04_comparacao_modelos.png`).

Uso nos slides:

> Este grafico mostra por que separamos as metricas em dois paineis. Como RMSE e MAE tem escala muito maior que R2, colocar tudo no mesmo eixo dificulta a interpretacao.

## Figuras recomendadas para os slides

Para a apresentacao, priorize:

1. `gpu_01_distribuicao_rendimento.png`
2. `gpu_02_matriz_correlacao.png`
3. `gpu_03_rendimento_por_cultura.png`
4. `gpu_04_comparacao_modelos.png`
5. `gpu_05_erros_modelos.png`
6. `gpu_06_r2_modelos.png`
7. `gpu_07_real_vs_predito_melhor_modelo.png`
8. `gpu_09_residuos_por_modelo.png`

Evite usar `gpu_14_comparacao_modelos_escala_unica.png`, porque ele mistura metricas em escalas muito diferentes.
