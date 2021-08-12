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

### Eu preciso usar todas essas colunas?
No total, a base de dados conta com 231 colunas, das quais:
* 13 eram variáveis pré-existentes dos pacientes;
* 216 variáveis relacionadas a exames;
* 1 variável que indicava a janela de tempo;
* 1 variável resposta.

Para facilitar a vida do nosso modelo, utilizou-se duas metodologias de *feature selection* ou *seleção de variáveis* nas colunas relacionadas a exames:
#### Variance threshold

#### Correlação alta entre colunas

## Machine learning

### Qual modelo eu escolho?

### Tunando o modelo escolhido



# Resultados e Discussão


# Conclusão

# Referências

1. https://www.paho.org/pt/covid19/historico-da-pandemia-covid-19

2. https://relief.unboundmedicine.com/relief/view/Coronavirus-Guidelines/2355041/all/Epidemic__Epi__Curves_for_Coronavirus_COVID_19
