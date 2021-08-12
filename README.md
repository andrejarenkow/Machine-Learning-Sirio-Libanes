![](https://github.com/decao88/Machine-Learning-Sirio-Libanes/blob/main/images/Hospital%20s%C3%ADrio-liban%C3%AAs.png?raw=true)

# Machine Learning Sírio-Libanês
*Projeto de finalização de curso*


# Introdução

## Quem sou eu?
Me chamo André Jarenkow, sou engenheiro químico formado na Universidade Federal do Rio Grande do Sul, possuo mestrado em Engenharia Química, e atualmente trabalho na Secretaria Estadual do Rio Grande do Sul, mais precisamente no [Centro Estadual de Vigilância em Saúde](https://www.cevs.rs.gov.br/inicial) em Porto Alegre, na Vigilância Ambiental. Gosto muito de trabalhar com planilhas, dados e ir futricando até conseguir tirar suco de pedra.

## O que é este projeto?
O projeto atual trata-se do trabalho final do Bootcamp de Data Science Aplicada da Alura. A ideia é propor um modelo de *Machine Learning* baseado no banco de dados disponibilizado pelo [Hospital Sírio-Libanês no Kaggle](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19), com o objetivo de prever se o paciente que era internado iria ou não para a Unidade de Tratamento Intensivo, com base nas suas características e exames realizados ao longo da sua internação clínica.

## COVID-19

No final de 2019, uma doença emergente na província de Wuhan começou a preocupar as autoriades de saúde pública, pois estava se espalhando rapidamente e poderia tomar proporções globais<sup>[1](https://www.paho.org/pt/covid19/historico-da-pandemia-covid-19)</sup>. Esta nova doença foi batizada de COVID-19, e era causada pelo coronavírus SARS-CoV-2. Em março de 2020, a Organização Mundial da Saúde declarou uma epidemia de níveis globais, também conhecida como *Pandemia*.

Apesar da taxa de mortalidade relativamente baixa, a COVID-19 pode levar o indivíduo à internação em Unidades de Terapia Intensiva (UTI), dependendo das comorbidades da pessoa diagnosticada com a doença. Uma preocupação muito pertinente das autoridades de saúde pública era a capacidade de atendimento do sistema de saúde, afinal, se muitas pessoas se infectassem, por mais que a porcentagem que precisa ir para o hospital seja baixa, 1% de 100.000 são 1.000 pessoas. Os hospitais possuem tantos leitos assim?

![](https://relief.unboundmedicine.com/relief/repview?type=730-2180&name=5_2355041_Standard)

Pensando nisso, o Hospital Sírio-Libanês disponibilizou uma base de dados anonimizada para que cientistas de dados amadores e profissionais ajudassem na elaboração de um modelo de *Machine Learning* que pudesse ajudar a prever a possibilidade de um paciente ir ou não para a UTI.

# Metodologia

## Tratamento dos dados
Apesar da base de dados já estar bastante trabalhada, alguns ajustes precisavam ser feitos a fim de encaixá-la no treinamento do modelo. Por exemplo, o banco de dados mostra mais de uma linha por paciente, mais precisamente são 5 linhas por paciente, cada uma mostrando a evolução do paciente dentro do hospital.

### O que fazer com *NaN*?
Um dos principais problemas em um banco de dados é quantidade de dados que são *NaN*, ou *Not a Number*, também conhecidos como *vazio do inferno*. A metodologia utilizada neste trabalho foi a de que os exames de cada paciente eram invariáveis enquanto o paciente estivesse internado. Foi utilizada então a função *fillna()*, com os parâmetros de *foward fill* (**ffill**) e *backfill* (**bfill**).

### Por que tem tanta linha aqui da mesma pessoa?
Como dito anteriormente, cada paciente possuía 5 linhas no banco de dados, uma para cada janela de tempo da permanência dele no hospital. O problema é que isso complica a vida do nosso modelo, pois queremos saber se o paciente vai ou não para a UTI baseado na sua admissão no hospital. Assim, utilizou-se somente primeira janela de tempo de cada paciente (0 a 2 horas). Para manter a coluna de internação em UTI, a qual é binária (0 ou 1), apenas passou-se uma tabela dinâmica de valor máximo desta coluna por paciente. 

### Como transformo *string* em número?

É interessante verificar que a coluna *AGE_PERCENTIL* é uma string, e teremos que transformá-la em numérica através da técnica de [colunas dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html), a qual vai criar uma nova coluna para cada valor único e marcará com 1 a coluna do valor que aquela linha possuía.

### Eu preciso usar todas essas colunas mesmo?
No total, a base de dados conta com 231 colunas, das quais:
* 13 eram variáveis pré-existentes dos pacientes;
* 216 variáveis relacionadas a exames;
* 1 variável que indicava a janela de tempo;
* 1 variável resposta.

Para facilitar a vida do nosso modelo, utilizou-se duas metodologias de *feature selection* ou *seleção de variáveis* nas colunas relacionadas a exames:
#### Variance threshold
Uma metodologia disponível dentro do pacote do [Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold) a qual avalia a variância dentro da coluna e, caso esta seja menor do que o valor setado, a coluna é retirada do banco. No estudo, foi utilizado como valor de corte apenas aquelas que não possuíam variação nenhuma.

#### Correlação alta entre colunas
Pensemos assim: digamos que eu quero prever se uma pessoa vai precisar usar o cartão de crédito em um determinado mês, e dentre as variáveis que eu vou utilizar para prever estão o número de roupas que esta pessoa comprou no mês e quantas vezes ela foi a uma loja de roupas. Conseguimos afirmar que estas duas variáveis são correlacionadas, pois quando uma aumenta, a outra também aumenta! Para o modelo, é quase que uma redundância, pois ele pode utilizar apenas uma delas para fazer a previsão.

No caso dos dados do Sírio-Libanês não é diferente, e algumas colunas podem ser cortadas! Criou-se então uma matriz de correlação entre estas colunas e retirou-se uma de cada par com correlação maior do que 0,95.

## Machine learning

### Qual modelo eu escolho?
Após a limpeza e tratamento dos dados, chegou a hora de testar modelos de classificação! Foram escolhidos os seguintes:
* [Suport Vector Machines](https://scikit-learn.org/stable/modules/svm.html#classification)
* [Decision Trees](https://scikit-learn.org/stable/modules/tree.html#classification)
* [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
* Como referência, [Dummy Classifier.](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier)

Para diminuir a aleatoriedade do modelo, utilizou-se a [*Cross validation*](https://scikit-learn.org/stable/modules/cross_validation.html) com [*Repeated KFold Stratified*](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html#sklearn.model_selection.RepeatedStratifiedKFold), o qual vai repetir n vezes o processo de validação, de maneira estratificada (dividida igualmente de acordo com os dados), dividindo-o em treino e teste n vezes e verificando a métrica em cada uma destas vezes.
Como métrica de avaliação, utilizou-se a média do *Receiver Operating Characteristic - Area Under the Curve*, ou também apelidado de ROC_AUC médio. A métrica foi verificada tanto nos dados de teste quanto nos dados de treino, a fim de visualizar o [*overfitting*](https://pt.wikipedia.org/wiki/Sobreajuste). 

### Tunando o modelo escolhido

Após a escolha do modelo, os hiperparâmetros foram escolhidos com a ajuda de um laço **FOR**, o qual rodou a mesma métrica para diferentes valores pré-estabelecidos dos parâmetros do modelo.

# Resultados e Discussão

Após o *feature selection*, o modelo foi treinado com 70 colunas, diminuindo 151 colunas.
Na tabela abaixo, é possível verificar como foram os diferentes resultados para métricas de ROC_AUC nos modelos testados:
Modelo | ROC_AUC médio - teste | ROC_AUC médio - treino
------------ | ------------- | --------------
SVM | 0,771 | 0,840
Random Forest - Ensemble Methods | 0,751 | 0,999
Decision Tree | 0,604 | 1,000
Dummy Classifier | 0,500 | 0,5000

Como é possível verificar, o modelo de SVM obteve os melhores resultados de ROC_AUC, tanto no teste, quanto no treino, não chegando a *overfitar*.

Os hiperparâmetros escolhidos para variar foram <sup>[3](https://scikit-learn.org/stable/modules/svm.html), [4](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC), [5](https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/)</sup> :

* **Kernel** -  *linear*, *poly*, *rbf* e *sigmoid*
* **C** - 1 e 10
* **gamma** - 1, 0,10 e 0,01.

Dessa vez, foram 24 combinações diferentes. Na tabela abaixo coloco os 5 primeiros colocados em relação ao ROC_AUC médio de teste:

Kernel | C | gamma | ROC_AUC médio - teste | ROC_AUC médio - treino
------------ | ------------- | -------------- | ---------------| ----------
rbf | 1 | 0,10 | 0,782 | 0,889
rbf | 10 | 0,01 | 0,771 | 0,856
poly | 10 | 0,01 | 0,767 | 0,808
rbf | 10 | 0,10 | 0,767 | 0,987
linear | 1 | 1 | 0,761 | 0,861

Olhando de maneira crua, é possível dizer que a combinação de kernel = rbf, C = 1 e gamma  = 0,10 foi a melhor, no entanto, o ROC_AUC de treino foi um pouco mais elevado também. Acredito que a escolha do segundo lugar seria uma opção tão boa quanto a primeira.

# Conclusão

A limpeza e tratamento dos dados é muito bem vinda, uma vez que facilitou a vida do nosso modelo, diminuindo o poder computacional necessário para que o mesmo seja treinado. Dentre os modelos testados, aquele que mais se aproximou de um valor ótimo foi o de Suport Vector Machines, com hiperparâmetros de Kernel = rbf, C = 1 e gamma = 0,10, atingindo um ROC_AUC médio de teste de 0,782. É um valor bastante razoável, levando em conta que o modelo não 

# Referências

1. https://www.paho.org/pt/covid19/historico-da-pandemia-covid-19
2. https://relief.unboundmedicine.com/relief/view/Coronavirus-Guidelines/2355041/all/Epidemic__Epi__Curves_for_Coronavirus_COVID_19
3. https://scikit-learn.org/stable/modules/svm.html
4. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
5. https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
