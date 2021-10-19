# Simple linear regression

#importando bibliotecas

import numpy as np; #computação dos dados
import matplotlib.pyplot as plt; # vizualização dos dados
import pandas as pd; # conjunto de dados e e o vetori variavel dependente

# data set ler todos os dados e criar o data frame
dataset = pd.read_csv('Data.csv') #iloc coleta os indices das linhas e colunas que queremos
X = dataset.iloc[:, :-1].values # iloc seleciona todas as colunas (:) exclui a ultima coluna (:-1)->pega no intervalo
Y = dataset.iloc[:, -1].values # de todas as linhas quero a ultima
# X = matriz de recurso Y = matriz dependente
from sklearn.impute import SimpleImputer #para completar valores ausentes

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #é apenas o objeto especifica que são dados vazios e o seg argumento é como você quer substiuilos
print(imputer)
#agora aplicamos esse objeto na matriz de recurso
imputer.fit(X[:, 1:3])#espera as colunas de X com valores numericos 
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
cTransfer = ColumnTransformer(transformers =[('encoder',OneHotEncoder(),[0])], remainder='passthrough') #Tipo de transformção que to codificado, tipo de codificamento que queremos fazer(quente ), os indices as colunas que queremos codificar
X = np.array(cTransfer.fit_transform(X))
print(X)

from sklearn.preprocessing import LabelEncoder
labelEn = LabelEncoder()
Y = labelEn.fit_transform(Y)
print(Y)

from sklearn.model_selection import train_test_split  #divide o dataset um tam especifico
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) 

print(X_train)
print(Y_train)
print(X_test)
print(Y_test)
#as decicoes de Ytrain vem correspodentes pela Xtrain

from sklearn.preprocessing import StandardScaler
sc  = StandardScaler() #não vai ser aplica
X_train[:,3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)