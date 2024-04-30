import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

dados = pd.read_csv('prouni_2005_2019.csv')

#print(dados.head())
#print(dados.shape)
#print(dados.dtypes)

y = dados['idade'].values
x= dados['ANO_CONCESSAO_BOLSA'].values


#Separando dados de treino e de teste
#utilizamos 70% dos dados para treino e o restante (30%) para teste.
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3)

#Precisamos redimensionar os dados para fazer a regressão linear
x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
y_test=y_test.reshape(-1,1)

#treinando o modelo
reg = LinearRegression()
reg.fit(x_train,y_train)
pred = reg.predict(x_test)

r = pearsonr(x, y)
print(f'Coeficiente de correlação: {r}')

plt.scatter(x, y, color="blue")
plt.plot(x_test, pred, color="red")
plt.title("IDADE X ANO_BOLSA")
plt.xlabel("IDADE")
plt.ylabel("ANO_CONCESSAO_BOLSA")
plt.show()

