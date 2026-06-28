# Explicacao dos modelos usados

Este documento explica, com mais profundidade, os tres modelos treinados no trabalho:

- MLP: Multi-Layer Perceptron.
- XGBoost: Extreme Gradient Boosting.
- CatBoost: Categorical Boosting.

Todos foram usados para uma tarefa de regressao supervisionada. Isso significa que o modelo aprende a prever um valor numerico, neste caso o rendimento agricola `hg/ha_yield`, a partir de exemplos ja conhecidos.

## Antes dos modelos: o que eles recebem

Cada modelo recebe uma tabela com atributos como:

- pais ou regiao (`Area`);
- cultura agricola (`Item`);
- ano (`Year`);
- chuva;
- temperatura;
- pesticidas;
- historico de rendimento.

O objetivo e aprender uma funcao aproximada:

```text
dados de entrada -> rendimento agricola previsto
```

Depois do treinamento, o modelo consegue receber uma nova combinacao de cultura, regiao, clima e historico, e devolver uma estimativa de rendimento.

## MLP - Multi-Layer Perceptron

MLP significa `Multi-Layer Perceptron`, ou Perceptron Multicamadas.

E uma rede neural artificial formada por camadas de neuronios. Cada neuronio faz uma pequena operacao matematica, e o conjunto das camadas aprende relacoes complexas entre as variaveis de entrada e o alvo.

### Como funciona em ideia simples

1. O modelo recebe os atributos numericos da amostra.
2. A primeira camada combina esses atributos com pesos matematicos.
3. Funcoes de ativacao, como ReLU, permitem aprender relacoes nao lineares.
4. Camadas intermediarias transformam os dados em representacoes internas.
5. A camada final gera um unico numero: o rendimento previsto.

No trabalho, a MLP foi implementada com PyTorch e treinada em GPU no Google Colab.

### Por que usar MLP neste problema

A produtividade agricola pode depender de relacoes nao lineares. Por exemplo, temperatura e chuva nao afetam a producao de forma sempre proporcional. Uma MLP consegue aprender combinacoes complexas entre variaveis como clima, cultura, regiao e historico.

### Pontos fortes

- Aprende padroes complexos.
- Funciona bem com muitas variaveis numericas.
- Pode capturar relacoes nao lineares.

### Limites

- E menos interpretavel que uma arvore de decisao.
- Precisa de normalizacao/escala dos dados para treinar melhor.
- Pode exigir mais tempo de treinamento.

### Resultado no trabalho

A MLP foi o melhor modelo por RMSE e R2:

```text
RMSE = 9225,74
MAE = 4116,18
R2 = 0,9883
```

Isso significa que, no experimento final, ela foi a melhor em erro geral e na capacidade de explicar a variacao do rendimento.

## XGBoost - Extreme Gradient Boosting

XGBoost significa `Extreme Gradient Boosting`.

Ele e um modelo baseado em varias arvores de decisao treinadas em sequencia. Cada nova arvore tenta corrigir parte dos erros cometidos pelas arvores anteriores.

### Como funciona em ideia simples

1. O modelo cria uma primeira arvore de decisao.
2. Essa arvore faz previsoes e comete erros.
3. Uma nova arvore e treinada para reduzir esses erros.
4. O processo se repete varias vezes.
5. A previsao final combina o resultado de muitas arvores.

Esse processo e chamado de boosting.

### Por que usar XGBoost neste problema

O XGBoost e muito usado em problemas tabulares, como planilhas com colunas numericas e categoricas transformadas. O dataset deste trabalho e exatamente desse tipo: uma tabela com informacoes de clima, cultura, regiao e ano.

### Pontos fortes

- Costuma ter desempenho alto em dados tabulares.
- Captura relacoes nao lineares.
- Lida bem com interacoes entre variaveis.
- E mais facil de explicar que uma rede neural, pois se baseia em arvores de decisao.

### Limites

- Pode ficar sensivel a hiperparametros.
- Muitas arvores podem aumentar o tempo de treinamento.
- Ainda nao e tao simples de interpretar quanto uma unica arvore de decisao.

### Resultado no trabalho

O XGBoost foi o melhor modelo por MAE:

```text
RMSE = 9362,52
MAE = 3643,92
R2 = 0,9879
```

Isso indica que ele teve o menor erro absoluto medio, mesmo nao sendo o melhor por RMSE e R2.

## CatBoost - Categorical Boosting

CatBoost significa `Categorical Boosting`.

Assim como o XGBoost, ele tambem e baseado em boosting com arvores de decisao. A diferenca principal e que o CatBoost foi projetado para lidar muito bem com variaveis categoricas.

No script deste trabalho, `Area` e `Item` foram transformadas com one-hot encoding antes do treinamento. Mesmo assim, o CatBoost continua sendo uma boa escolha para comparacao porque e um modelo forte para dados tabulares.

### Como funciona em ideia simples

1. O CatBoost treina arvores de decisao em sequencia.
2. Cada arvore tenta corrigir os erros acumulados.
3. O algoritmo usa estrategias para reduzir overfitting.
4. A previsao final combina muitas arvores.

### Por que usar CatBoost neste problema

O dataset tem variaveis categoricas importantes, principalmente `Area` e `Item`. Como o CatBoost foi criado pensando nesse tipo de dado, ele e um bom candidato para prever rendimento agricola em uma base com paises/regioes e culturas.

### Pontos fortes

- Muito forte em dados tabulares.
- Bom para problemas com variaveis categoricas.
- Geralmente exige menos ajuste manual que outros modelos de boosting.
- Tem suporte a GPU.

### Limites

- Pode ser mais pesado para instalar e treinar.
- Ainda exige cuidado com validacao para evitar conclusoes exageradas.
- No trabalho, ficou ligeiramente abaixo da MLP e do XGBoost nas principais metricas.

### Resultado no trabalho

O CatBoost teve desempenho muito proximo dos outros modelos:

```text
RMSE = 9409,89
MAE = 3777,14
R2 = 0,9878
```

Isso mostra que ele tambem aprendeu bem os padroes do dataset.

## Comparacao entre os modelos

| Modelo | Ideia central | Melhor ponto | Cuidado |
|---|---|---|---|
| MLP | Rede neural com camadas | Melhor RMSE e R2 | Menos interpretavel |
| XGBoost | Muitas arvores corrigindo erros anteriores | Melhor MAE | Requer ajuste de hiperparametros |
| CatBoost | Boosting com foco em dados categoricos | Forte para dados tabulares/categoricos | Pode ser mais pesado |

## Qual e mais facil de explicar na apresentacao

O mais facil de explicar e o XGBoost, porque a ideia de arvores de decisao e intuitiva:

> O modelo usa varias arvores. Cada arvore tenta melhorar os erros das anteriores, e no final todas contribuem para uma previsao mais precisa.

O CatBoost pode ser explicado logo depois, como um modelo parecido com XGBoost, mas especialmente bom para dados com categorias.

A MLP exige um pouco mais de cuidado, porque envolve redes neurais:

> A MLP aprende combinacoes matematicas entre as variaveis em varias camadas, tentando aproximar a relacao entre clima, cultura, historico e rendimento.

## Relacao com o Estudo X

O Estudo X comparou modelos de aprendizado de maquina para previsao de rendimento agricola e reportou bons resultados para modelos baseados em arvores, como Random Forest, Bagging Regressor e XGBoost.

Neste trabalho, usamos XGBoost como ponto de ligacao direta com o estudo, e adicionamos MLP e CatBoost para comparar tecnicas modernas tambem adequadas para dados tabulares.

A comparacao principal foi feita com R2, porque essa metrica aparece tanto nos resultados deste trabalho quanto nos resultados reportados pelo Estudo X.

## Frase curta para os slides

> Foram comparados uma rede neural MLP e dois modelos de boosting baseados em arvores, XGBoost e CatBoost. A MLP teve o melhor RMSE e R2, enquanto o XGBoost teve o menor MAE. Todos apresentaram desempenho alto, com R2 acima de 0,987.
