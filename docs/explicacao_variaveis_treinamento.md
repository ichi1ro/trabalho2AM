# Explicacao das variaveis usadas no treinamento

O dataset usado foi o Kaggle `patelris/crop-yield-prediction-dataset`. Entre os CSVs baixados, o trabalho usa principalmente o arquivo `yield_df.csv`, que ja combina informacoes de producao agricola, chuva, temperatura e pesticidas.

## Objetivo da previsao

O alvo do modelo e:

| Variavel | Tipo | Explicacao |
|---|---|---|
| `hg/ha_yield` | numerica | Rendimento agricola da cultura, medido em hectogramas por hectare. E o valor que os modelos tentam prever. |

Os modelos recebem informacoes sobre pais/regiao, cultura, ano, clima, manejo e historico produtivo, e retornam uma estimativa de rendimento agricola.

## Variaveis originais usadas

Estas variaveis vieram diretamente do dataset consolidado.

| Variavel | Tipo | Papel no modelo | Explicacao curta |
|---|---|---|---|
| `Area` | categorica | identificacao geografica | Pais ou regiao onde a cultura foi registrada. |
| `Item` | categorica | identificacao da cultura | Tipo de cultura agricola, como milho, arroz, soja etc. |
| `Year` | numerica | contexto temporal | Ano da observacao. Ajuda o modelo a capturar mudancas ao longo do tempo. |
| `average_rain_fall_mm_per_year` | numerica | clima | Chuva media anual, em milimetros por ano. |
| `pesticides_tonnes` | numerica | manejo agricola | Quantidade de pesticidas usada, em toneladas. Funciona como uma aproximacao de intensidade de manejo. |
| `avg_temp` | numerica | clima | Temperatura media, em graus Celsius. |

## Variaveis criadas pelo trabalho

Estas variaveis nao vieram prontas no dataset. Elas foram criadas no script para dar mais contexto aos modelos.

| Variavel | Como foi criada | Explicacao |
|---|---|---|
| `temp_squared` | `avg_temp` ao quadrado | Ajuda o modelo a capturar efeito nao linear da temperatura. Temperaturas muito baixas ou muito altas podem afetar a produtividade de formas diferentes. |
| `rain_temp_interaction` | chuva anual multiplicada pela temperatura media | Representa o efeito combinado de chuva e temperatura. A mesma quantidade de chuva pode ter impacto diferente em regioes mais frias ou mais quentes. |
| `yield_lag_1` | rendimento do ano anterior para a mesma `Area` e `Item` | Da memoria historica ao modelo. Se uma cultura teve alto rendimento em uma regiao no ano anterior, isso pode ser informativo. |
| `yield_rolling_mean_3` | media do rendimento dos ultimos ate 3 anos para a mesma `Area` e `Item` | Suaviza os dados historicos e mostra uma tendencia recente da produtividade. |

## Lista final de atributos

Os 10 atributos finais usados para prever `hg/ha_yield` foram:

1. `Area`
2. `Item`
3. `Year`
4. `average_rain_fall_mm_per_year`
5. `pesticides_tonnes`
6. `avg_temp`
7. `temp_squared`
8. `rain_temp_interaction`
9. `yield_lag_1`
10. `yield_rolling_mean_3`

## Como as variaveis entram nos modelos

As variaveis categoricas `Area` e `Item` nao entram como texto puro. O script transforma essas colunas com one-hot encoding.

Exemplo simplificado:

```text
Item = Maize
```

vira algo como:

```text
Item_Maize = 1
Item_Rice = 0
Item_Soybeans = 0
```

Isso permite que os modelos matematicos usem informacoes categoricas sem tratar os nomes como numeros arbitrarios.

## Por que ordenar por Area, Item e Year

Antes de criar `yield_lag_1` e `yield_rolling_mean_3`, o script ordena os dados por:

```text
Area, Item, Year
```

Essa ordenacao e necessaria porque as variaveis historicas dependem da sequencia temporal correta. O rendimento anterior de milho no Brasil, por exemplo, deve ser calculado usando anos anteriores do proprio milho no Brasil, e nao de outra cultura ou regiao.

## Variaveis que mais ajudam a ideia do trabalho

As variaveis podem ser entendidas em quatro grupos:

| Grupo | Variaveis | Ideia |
|---|---|---|
| Identificacao | `Area`, `Item`, `Year` | Diz onde, o que e quando esta sendo previsto. |
| Clima | `average_rain_fall_mm_per_year`, `avg_temp` | Representa condicoes ambientais. |
| Manejo | `pesticides_tonnes` | Aproxima a intensidade de intervencao agricola. |
| Historico | `yield_lag_1`, `yield_rolling_mean_3` | Usa memoria produtiva da cultura naquela regiao. |

## Cuidado na interpretacao

As variaveis historicas sao muito fortes. Isso ajuda os modelos a terem bom desempenho, mas tambem exige cuidado: parte da previsao pode estar vindo da repeticao de padroes historicos, e nao apenas do efeito direto do clima ou do manejo.

Por isso, nos slides, a interpretacao correta e:

> O modelo usa informacoes de regiao, cultura, clima, manejo e historico para estimar rendimento agricola. Ele apoia a tomada de decisao, mas nao substitui avaliacao agronomica local.
