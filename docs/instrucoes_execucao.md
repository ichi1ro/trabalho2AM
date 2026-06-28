# Instrucoes de execucao no Google Colab

Passo a passo para reproduzir o trabalho no Google Colab usando o notebook principal e GPU.

## 1. Abrir o notebook no Colab

1. Abra `notebooks/trabalho_agro_previsao_safra.ipynb` no Google Colab.
2. Execute as celulas na ordem em que aparecem.
3. No menu, selecione `Runtime > Change runtime type`.
4. Em `Hardware accelerator`, escolha `T4 GPU`.
5. Salve a configuracao.

O notebook ja contem a celula responsavel por preparar o acesso ao repositorio. Nao e necessario adicionar manualmente um passo de clonagem nas instrucoes.

Essa configuracao de GPU e importante porque o script final foi preparado para rodar MLP, XGBoost e CatBoost com aceleracao de hardware.

## 2. Instalar as bibliotecas

Execute a celula de instalacao do notebook. O comando usado e:

```python
!pip install -r requirements-colab.txt
```

As principais bibliotecas usadas sao:

- `pandas` e `numpy` para manipulacao dos dados.
- `scikit-learn` para metricas, validacao cruzada e pre-processamento.
- `matplotlib` e `seaborn` para graficos.
- `xgboost` para o modelo XGBoost.
- `catboost` para o modelo CatBoost.
- `torch` para a MLP em PyTorch.
- `kagglehub` para baixar o dataset do Kaggle.

## 3. Baixar o dataset do Kaggle

Execute a celula de download. O comando usado e:

```python
!python src/download_data.py
```

O script baixa o dataset:

```text
patelris/crop-yield-prediction-dataset
```

e copia os CSVs para:

```text
data/raw
```

O treinamento usa principalmente:

```text
data/raw/yield_df.csv
```

## 4. Rodar uma execucao rapida de teste

Antes da execucao final, rode uma versao menor para confirmar se o ambiente esta funcionando:

```python
!python src/train_evaluate_gpu_colab.py --folds 3 --sample-size 6000
```

Essa versao usa uma amostra menor e menos folds. Ela serve apenas para teste tecnico, nao para os resultados finais do trabalho.

## 5. Rodar a execucao final

Depois do teste, rode:

```python
!python src/train_evaluate_gpu_colab.py --folds 5
```

Essa execucao:

- carrega o dataset;
- remove a coluna de indice desnecessaria, se existir;
- ordena os dados por `Area`, `Item` e `Year`;
- cria variaveis derivadas;
- aplica one-hot encoding em `Area` e `Item`;
- treina MLP, XGBoost e CatBoost;
- avalia os modelos com validacao cruzada k-fold;
- salva metricas, predicoes e figuras.

## 6. Saidas geradas

Depois da execucao, os arquivos principais ficam em:

```text
outputs/results_gpu
outputs/figures_gpu
```

Arquivos de resultados:

- `outputs/results_gpu/model_results_gpu.csv`
- `outputs/results_gpu/predictions_gpu.csv`
- `outputs/results_gpu/resultados_gpu.md`

Figuras geradas:

- `outputs/figures_gpu/gpu_01_distribuicao_rendimento.png`
- `outputs/figures_gpu/gpu_02_matriz_correlacao.png`
- `outputs/figures_gpu/gpu_03_rendimento_por_cultura.png`
- `outputs/figures_gpu/gpu_04_comparacao_modelos.png`
- `outputs/figures_gpu/gpu_05_erros_modelos.png`
- `outputs/figures_gpu/gpu_06_r2_modelos.png`
- `outputs/figures_gpu/gpu_07_real_vs_predito_melhor_modelo.png`
- `outputs/figures_gpu/gpu_08_residuos_melhor_modelo.png`
- `outputs/figures_gpu/gpu_09_residuos_por_modelo.png`
- `outputs/figures_gpu/gpu_10_real_vs_predito_por_modelo.png`
- `outputs/figures_gpu/gpu_11_rendimento_medio_por_ano.png`
- `outputs/figures_gpu/gpu_12_chuva_vs_rendimento.png`
- `outputs/figures_gpu/gpu_13_temperatura_vs_rendimento.png`
- `outputs/figures_gpu/gpu_14_comparacao_modelos_escala_unica.png`

## 7. Compactar e baixar os resultados

No Colab, execute:

```python
!zip -r resultados_trabalho2.zip outputs data/processed
```

Depois:

```python
from google.colab import files
files.download("resultados_trabalho2.zip")
```

Esse arquivo compactado pode ser usado para atualizar slides e documentos locais.

## 8. Resultados finais de referencia

A execucao final ja realizada gerou estes resultados:

| Modelo | RMSE | MAE | R2 |
|---|---:|---:|---:|
| MLP | 9225,74 | 4116,18 | 0,9883 |
| XGBoost | 9362,52 | 3643,92 | 0,9879 |
| CatBoost | 9409,89 | 3777,14 | 0,9878 |

Melhor modelo por RMSE e R2: MLP.

Melhor modelo por MAE: XGBoost.

## 9. Observacao sobre nomes dos modelos

Nos arquivos gerados pelo Colab, os modelos aparecem com nomes tecnicos:

| Nome no arquivo | Nome para usar nos slides |
|---|---|
| `MLP_PyTorch_CUDA` | MLP |
| `XGBoost_GPU` | XGBoost |
| `CatBoost_GPU` | CatBoost |

O sufixo `GPU` ou `CUDA` indica apenas que o treinamento foi executado com aceleracao de hardware. Nos slides, nao e necessario manter esse sufixo.

## 10. Problemas comuns

### CUDA nao disponivel

Se aparecer erro dizendo que CUDA nao esta disponivel, confira se o Colab esta com GPU:

```python
import torch
print(torch.cuda.is_available())
```

O resultado esperado e:

```text
True
```

Se vier `False`, volte em `Runtime > Change runtime type` e selecione `T4 GPU`.

### Execucao muito demorada

Use primeiro a execucao reduzida:

```python
!python src/train_evaluate_gpu_colab.py --folds 3 --sample-size 6000
```

Depois rode a execucao final com:

```python
!python src/train_evaluate_gpu_colab.py --folds 5
```
