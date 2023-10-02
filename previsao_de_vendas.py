# ENTENDIMENTO DO DESAFIO
# ENTENDIMENTO DA ÁREA/EMPRESA


# EXTRAÇÃO/OBTENÇÃO DE DADOS
# importar a base de dados

import pandas as pd

tabela = pd.read_csv(r"db\advertising.csv")


print(tabela)



# AJUSTE DE DADOS (TRATAMENTO/LIMPEZA)

# ANÁLISE EXPLORATORIA
# analisar a tabela atraves de correlação 
import matplotlib.pyplot as plt     # -> biblioteca antiga
import seaborn as sns       # -> usa o matplotlib como base

print(tabela.corr())

# criar um grafico
sns.heatmap(tabela.corr(), cmap='Blues', annot=True)



# exibir o grafico

plt.show()

# MODELAGEM + ALGORITMOS (AQUI ENTRA A INTELIGENCIA ARTIFICIAL, SE NECESSARIO)

# separando em dado de treino e dados de teste

y = tabela["Vendas"]

x = tabela[["TV", "Radio", "Jornal"]]


from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

# Formar de treinar a inteligencia artificial

# REGRESSÃO LINEAR

# RANDOMFOREST (ARVORE DE DECISAO)


# importar a inteligencia artificial

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# criar a inteligencia artificial

modelo_regreaolinear = LinearRegression()

modelo_arvoredecisao = RandomForestRegressor()


# treinar a inteligencia artificial

modelo_regreaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

previsao_regressaolinear = modelo_regreaolinear.predict(x_teste)

previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

from sklearn.metrics import r2_score

print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvoredecisao))




# VISUALISAR AS PREVISOES


tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsao arvore decisao"] = previsao_arvoredecisao
tabela_auxiliar["previsao regresao Linear"] = previsao_regressaolinear
#print(tabela_auxiliar)

sns.lineplot(data=tabela_auxiliar)
#plt.show()



# FAZER UMA PREVISÃO

nova_tabela = pd.read_csv(r"db\novos.csv")

print(nova_tabela)


previsao = modelo_arvoredecisao.predict(nova_tabela)

print(previsao)


# INTERPRETAÇAO DE RESULTADOS
# 0   23.1    3.8    69.2
# 1   44.5    0.0     5.1
# 2  170.2   45.9     0.0
# [ 7.742  8.827 20.161]

# 7.742 -> previsao de venda caso gaste 23.1, 3.8, 69.2
# 8.827 -> previsao de venda caso gaste 44.5, 0.0, 5.1
# 20.161 -> previsao de venda caso gaste 170.2, 45.9, 0.0
