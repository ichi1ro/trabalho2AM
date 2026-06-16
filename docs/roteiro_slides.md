# Roteiro de slides

Sugestao para cerca de 15 slides.

## Slide 1 - Titulo

Previsao de rendimento agricola com aprendizado de maquina.

Imagem sugerida: foto de lavoura ou grafico `05_real_vs_predito_melhor_modelo.png`.

## Slide 2 - Problema

Explicar que prever rendimento agricola ajuda planejamento de safra, gestao de insumos e tomada de decisao no agronegocio.

## Slide 3 - Objetivo

Construir modelos de regressao para prever `hg/ha_yield` usando dados climaticos, historicos e de cultura.

## Slide 4 - Dataset

Dataset Kaggle: `patelris/crop-yield-prediction-dataset`.

Mostrar: 28.242 registros, periodo 1990-2013, paises/regioes, culturas, chuva, pesticidas, temperatura e rendimento.

## Slide 5 - Estudo X

Yan et al. (2025), "Crop Yield Time-Series Data Prediction Based on Multiple Hybrid Machine Learning Models".

Resultados do artigo: RF R2 = 0,986; Bagging R2 = 0,986; XGBoost R2 = 0,973.

## Slide 6 - Analise exploratoria

Usar `01_distribuicao_rendimento.png`.

Comentar que o rendimento tem grande variacao entre culturas e regioes.

## Slide 7 - Correlacao

Usar `02_matriz_correlacao.png`.

Destacar relacao entre variaveis historicas de rendimento e alvo.

## Slide 8 - Cultura agricola

Usar `03_rendimento_por_cultura.png`.

Comentar que culturas diferentes possuem escalas produtivas distintas.

## Slide 9 - Atributos finais

Listar os 10 atributos finais: `Area`, `Item`, `Year`, chuva, pesticidas, temperatura, temperatura ao quadrado, interacao chuva-temperatura, rendimento anterior e media movel.

## Slide 10 - Pipeline

Mostrar fluxo: download, limpeza, engenharia de atributos, preprocessamento, grid search, k-fold e avaliacao.

Imagem sugerida: criar um fluxograma simples no PowerPoint/Canva com essas etapas.

## Slide 11 - Modelos

SVM LinearSVR, Random Forest e MLP.

Explicar em uma frase o papel de cada modelo.

## Slide 12 - Resultados

Usar `04_comparacao_modelos.png`.

Tabela:

| Modelo | RMSE | MAE | R2 |
|---|---:|---:|---:|
| SVM LinearSVR | 10202,52 | 3108,02 | 0,9861 |
| Random Forest | 10735,80 | 3976,03 | 0,9847 |
| MLP | 11899,04 | 6157,89 | 0,9812 |

## Slide 13 - Real vs predito

Usar `05_real_vs_predito_melhor_modelo.png`.

Interpretar: pontos proximos da diagonal indicam boas previsoes.

## Slide 14 - Importancia dos atributos

Usar `06_importancia_atributos.png`.

Destacar `yield_lag_1` como principal atributo, mostrando que historico produtivo e decisivo.

## Slide 15 - Conclusao

O trabalho cumpriu o pipeline completo de AM. O melhor modelo local foi SVM LinearSVR com R2 = 0,9861. O desempenho em R2 ficou proximo do Estudo X, mas a comparacao precisa considerar que o modo local usou amostra reduzida. Trabalhos futuros: rodar modo completo no Colab, testar XGBoost e adicionar dados de solo ou manejo.
