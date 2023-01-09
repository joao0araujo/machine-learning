from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
import plotly.express as px

dadosE = pd.read_csv("https://raw.githubusercontent.com/joao0araujo/machine-learning/main/enemBA.csv")

dadosE['TP_SEXO'] = dadosE['TP_SEXO'].map({'M': 0, 'F': 1})

dadosE['Q002'] = dadosE['Q002'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7})

dadosE['Q006'] = dadosE['Q006'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16})

dadosE['Q024'] = dadosE['Q024'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4})

dadosE['Q022'] = dadosE['Q022'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4})

dadosE['Q025'] = dadosE['Q025'].map({'A': 0, 'B': 1})

dadosE['Q021'] = dadosE['Q021'].map({'A': 0, 'B': 1})

dadosE = dadosE.dropna(subset=['NU_NOTA_LC'])

dadosE = dadosE.dropna(subset= ['TP_ENSINO'])

dadosE = dadosE.dropna(subset= ['Q002'])

dadosE = dadosE.dropna(subset= ['Q006'])

dadosE = dadosE.dropna(subset= ['Q024'])

dadosE = dadosE.dropna(subset= ['Q025'])

X = dadosE[['TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_COR_RACA', 'TP_ESCOLA', 'TP_DEPENDENCIA_ADM_ESC', 'TP_LOCALIZACAO_ESC',
'TP_ENSINO', 'Q002', 'Q006', 'Q024', 'Q025', 'Q021', 'Q022']]
y = dadosE['NU_NOTA_LC']

#normalizando os valores numa escala 0 a 1 = melhora a perfomance do algoritmo
from sklearn.preprocessing import MinMaxScaler   # Serve para preprocessar os dados e escalar

normalizador = MinMaxScaler()
X_norm = normalizador.fit_transform(X)
X_norm

xtreino, xteste, ytreino, yteste = train_test_split(X_norm, y, test_size= 0.50, random_state= 10)

for i in range (30,50):

  knn =  KNeighborsRegressor(n_neighbors=i)  #instancia o modelo e o numero de vizinhos
  knn.fit(xtreino,ytreino)  #treina o modelo


  preditos  = knn.predict(xteste)  #vamos pegar os valores preditos com o modelo

  eqm = mean_squared_error(yteste, preditos)

  print('EQM da Regressao KNN ', i,' vizinhos -  {}'.format(eqm,2))



