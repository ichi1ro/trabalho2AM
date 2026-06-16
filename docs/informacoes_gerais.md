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

## Tecnicas de AM

- SVM LinearSVR.
- Random Forest Regressor.
- MLP Regressor.

## Metricas

- RMSE.
- MAE.
- R2.

## Validacao

- Modo rapido/local: 3-fold externo e 2-fold interno no grid search.
- Modo completo: 5-fold externo e 3-fold interno no grid search.

## Observacao importante

O computador local ficou lento com a configuracao completa. Por isso, o script agora usa modo rapido por padrao. A versao completa deve ser rodada no Google Colab ou em uma maquina com mais recursos.
