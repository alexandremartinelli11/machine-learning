
<style>
    table {
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 14px;
    text-align: center;
    }
    table td, table th {
    padding: 8px 12px;
    }
</style>

# Entrega Individual 

1. [Alexandre Martinelli](https://github.com/alexandremartinelli11){:target='_blank'}


## Introdução
1. Exploração de dados: Ao selecionar uma base no [kaggle](https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees){:target='_blank'} referentes a cinco tipos de remédio, remédio A, B, C, X e Y, tem como objetivo prever qual remédio o paciente teria uma resposta melhor. As colunas presentes nesse dataset são Idade, Sexo, Pressão Arterial, nivel de colesterol, nivel de sódio para potássio no sangue e remédio que seria nossa target. 

### Colunas
1. Age (Idade): Essa coluna temos a idade dos pacientes, com a idade minima presente de 15, idade média de 44,3 e maxima de 74 sendo do tipo Integer. 
2. Sex (Sexo): Essa coluna tem o sexo de cada paciente, divididos em 52% Masculino e 48% feminino, dados do tipo String.
3. Blood Pressure (Pressão Arterial): Essa coluna tem os niveis de pressão arterial de cada paciente sendo dividida em 39% HIGH, 29% NORMAL e 32% LOW, dados do tipo String.
4. Cholesterol (nivel de colesterol): Essa coluna tem os niveis de colesterol de cada paciente sendo divididos em 52% HIGH e 49% NORMAl, dados do tipo String.
5. Na_to_K (sódio para potássio): Essa coluna tem os a razão de sódio para potássio no sangue de um paciente, com a minima de 6,27, media de 16,1 e maxima de 38,2, dados do tipo Float/Decimal.
6. Drug (remédio): Essa coluna tem os remédio de melhor resposta para o paciente, dados do tipo String.

=== "Base"

    ```python exec="on" html="0"
    --8<-- "docs/arvore/original.py"
    ```


## Pré-processamento
Primeiro foi feita uma verificação em todas as colunas procurando valores faltantes e substituindo eles pela mediana em valores numéricos ou pela moda em variáveis categóricas. Como vimos na descrição das colunas temos três que possuem dados categóricos do tipo String, sendo elas Sex(Sexo), Blood Pressure(Pressão Arterial) e Cholesterol(nivel de colesterol), para conseguirmos utilizar essas informações é necessario convertelas em numeros, oque foi feito utilizando a biblioteca scikit-learn que possui a função LabelEncoder(), em seguida aplicamos dois tipos de escalonamento às colunas numéricas Age e Na_to_K: padronização (z-score) e normalização min-max.
 
=== "Result"

    ```python exec="on" html="0"
    --8<-- "docs/arvore/final.py"
    ```
=== "Prep Code"

    ```python
    --8<-- "docs/arvore/pre.py"
    ```
=== "Standardization"

    ```python exec="on"
    --8<-- "docs/arvore/normalizacao.py"
    ```

=== "Standardization code"

    ```python
    --8<-- "docs/arvore/normalizacao.py"
    ```

## Divisão dos Dados

Os dados foram divididos em 70% para treino e 30% para validação.


## Treinamento do Modelo

=== "Result 80% 20%"

    ```python exec="on" html="1"
    --8<-- "docs/arvore/arvore82.py"
    ```
=== "Result 70% 30%"

    ```python exec="on" html="1"
    --8<-- "docs/arvore/arvore73.py"
    ```
=== "Result 60% 40%"

    ```python exec="on" html="1"
    --8<-- "docs/arvore/arvore64.py"
    ```
=== "Result 50% 50%"

    ```python exec="on" html="1"
    --8<-- "docs/arvore/arvore55.py"
    ```
=== "Result 40% 60%"

    ```python exec="on" html="1"
    --8<-- "docs/arvore/arvore46.py"
    ```


## Referências

[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/reference/){:target='_blank'}