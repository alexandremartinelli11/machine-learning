
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
4. Cholesterol (nivel de colesterol): Essa coluna tem os niveis de colesterol de cada paciente sendo divididos em 52% HIGH e 49% NORMAL, dados do tipo String.
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

O conjunto de dados foi dividido em 70% para treino e 30% para validação, garantindo que o modelo fosse treinado em uma parte significativa das observações, mas ainda avaliado em dados não vistos. O uso do conjunto de validação tem como objetivo detectar e reduzir o risco de overfitting.


## Treinamento do Modelo

=== "Result 70% 30%"

    ```python exec="on" html="1"
    --8<-- "docs/svm/svm.py"
    ```

=== "Code"

    ```python exec="0" html="0"
    --8<-- "docs/svm/svm.py"
    ```


## Avaliação do Modelo

Comparando os diferentes kernels, o **linear** demonstrou ser a abordagem mais eficaz para este problema, atingindo **95% de acurácia** e superando os kernels sigmoid (88,3%), rbf (88,3%) e poly (85,0%). A visualização das fronteiras de decisão confirma que, após o pré-processamento, os dados se organizam de maneira que hiperplanos lineares são suficientes e eficientes para a separação das classes, evitando a complexidade desnecessária de kernels não lineares.

A análise detalhada pelo relatório de classificação aponta um desempenho robusto:
* **Performance Ideal:** A classe **DrugC (2)** foi classificada com perfeição (100% de precisão e recall).
* **Classes Majoritárias:** As classes **DrugX (3)** e **DrugY (4)** apresentaram resultados excelentes, com F1-Scores superiores a 0.96, demonstrando a capacidade do modelo em generalizar bem para os casos mais frequentes.
* **Análise de Erros:** A matriz de confusão revela que as poucas classificações incorretas tiveram um padrão específico. A classe **DrugA (0)** atuou como um "falso atrator", absorvendo incorretamente uma instância de DrugB e uma de DrugY. Isso resultou em um recall perfeito para DrugA, mas reduziu sua precisão para 78%.

Em conclusão, o SVM linear oferece o melhor equilíbrio entre viés e variância para este conjunto de dados, entregando uma solução simples, interpretável e com alta taxa de acerto global.



## Referências

[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/reference/){:target='_blank'}