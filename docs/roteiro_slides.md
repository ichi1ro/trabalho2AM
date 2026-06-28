# Roteiro de slides

Sugestao para cerca de 15 slides.

## Slide 1 - Titulo

Previsao de rendimento agricola com aprendizado de maquina.

Imagem sugerida: foto de lavoura ou grafico `gpu_07_real_vs_predito_melhor_modelo.png`.

## Slide 2 - Problema

Explicar que prever rendimento agricola ajuda planejamento de safra, gestao de insumos e tomada de decisao no agronegocio.

## Slide 3 - Objetivo

Construir modelos de regressao para prever `hg/ha_yield` usando dados climaticos, historicos e de cultura.

## Slide 4 - Dataset

Dataset Kaggle: `patelris/crop-yield-prediction-dataset`.

Mostrar: 28.242 registros brutos, periodo 1990-2013, paises/regioes, culturas, chuva, pesticidas, temperatura e rendimento.

## Slide 5 - Estudo X

Yan et al. (2025), "Crop Yield Time-Series Data Prediction Based on Multiple Hybrid Machine Learning Models".

Resultados do artigo:

| Modelo no Estudo X | Resultado reportado |
|---|---:|
| Random Forest | R2 = 0,986 |
| Bagging Regressor | R2 = 0,986 |
| XGBoost | R2 = 0,973 |

## Slide 6 - Analise exploratoria

Usar `gpu_01_distribuicao_rendimento.png`.

Comentar que o rendimento tem grande variacao entre culturas e regioes.

## Slide 7 - Correlacao

Usar `gpu_02_matriz_correlacao.png`.

Destacar relacao entre variaveis historicas de rendimento e alvo.

## Slide 8 - Cultura agricola

Usar `gpu_03_rendimento_por_cultura.png`.

Comentar que culturas diferentes possuem escalas produtivas distintas.

## Slide 9 - Atributos finais

Listar os 10 atributos finais: `Area`, `Item`, `Year`, chuva, pesticidas, temperatura, temperatura ao quadrado, interacao chuva-temperatura, rendimento anterior e media movel.

## Slide 10 - Pipeline

Mostrar fluxo: download, limpeza, engenharia de atributos, preprocessamento, busca de hiperparametros, k-fold e avaliacao.

Imagem sugerida: criar um fluxograma simples no PowerPoint/Canva com essas etapas.

## Slide 11 - Modelos

MLP, XGBoost e CatBoost.

| Modelo | Explicacao simples |
|---|---|
| MLP | E uma rede neural. Ela aprende relacoes nao lineares entre as variaveis de entrada e o rendimento agricola. |
| XGBoost | Usa arvores em sequencia. Cada nova arvore tenta corrigir os erros das arvores anteriores. Costuma ter desempenho muito forte em tabelas. |
| CatBoost | Tambem e um modelo de boosting com arvores. Ele e parecido com o XGBoost na ideia geral e costuma funcionar bem em dados tabulares. |

Mensagem principal:

> Escolhemos uma rede neural e dois modelos fortes de boosting com arvores. Assim comparamos uma abordagem neural com modelos de arvores muito usados em dados tabulares.

## Slide 12 - Resultados

Usar `gpu_04_comparacao_modelos.png`, `gpu_05_erros_modelos.png` e `gpu_06_r2_modelos.png`.

Tabela final:

| Modelo | RMSE | MAE | R2 |
|---|---:|---:|---:|
| MLP | 9225,74 | 4116,18 | 0,9883 |
| XGBoost | 9362,52 | 3643,92 | 0,9879 |
| CatBoost | 9409,89 | 3777,14 | 0,9878 |

Mensagem principal:

> Avaliamos os modelos por RMSE, MAE e R2. RMSE e MAE medem erro, entao quanto menor melhor. R2 mede quanto da variacao do rendimento foi explicada, entao quanto mais perto de 1 melhor.

Interpretacao do resultado:

> A MLP teve o melhor RMSE e o maior R2, entao foi o melhor modelo geral. O XGBoost teve o menor MAE, mostrando que, em media, seus erros absolutos foram menores. Os tres modelos ficaram muito proximos e com R2 acima de 0,987.

## Slide 13 - Real vs predito

Usar `gpu_07_real_vs_predito_melhor_modelo.png` ou `gpu_10_real_vs_predito_por_modelo.png`.

Interpretar: pontos proximos da diagonal indicam boas previsoes.

## Slide 14 - Erros e residuos

Usar `gpu_08_residuos_melhor_modelo.png` e `gpu_09_residuos_por_modelo.png`.

Explicacao:

> O residuo e a diferenca entre o valor previsto e o valor real. Residuos concentrados perto de zero indicam que o modelo errou pouco na maior parte dos casos.

## Slide 15 - Conclusao

O trabalho cumpriu o pipeline completo de AM: dataset do Kaggle, selecao de 10 variaveis, tres modelos, validacao cruzada, busca de hiperparametros, metricas e comparacao com Estudo X.

Fechamento:

> O melhor modelo foi a MLP, com R2 de 0,9883. Esse valor ficou acima do R2 de 0,986 reportado para Random Forest e Bagging no Estudo X, embora a comparacao precise considerar diferencas de protocolo experimental. Como melhoria futura, poderiamos adicionar dados de solo, manejo agricola e regioes mais detalhadas.
