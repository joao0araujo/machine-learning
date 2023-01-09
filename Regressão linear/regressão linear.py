from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
import plotly.express as px
import numpy as np

dadosE = pd.read_csv("https://raw.githubusercontent.com/joao0araujo/machine-learning/main/enemBA.csv")

#print(dadosE[['TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_COR_RACA', 'TP_ESCOLA', 'TP_ENSINO',
# 'TP_DEPENDENCIA_ADM_ESC', 'TP_LOCALIZACAO_ESC', 'Q002', 'Q006', 'Q024', 'Q025']])

dadosE['TP_SEXO'] = dadosE['TP_SEXO'].map({'M': 0, 'F': 1})

dadosE['Q002'] = dadosE['Q002'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7})

dadosE['Q006'] = dadosE['Q006'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16})

dadosE['Q024'] = dadosE['Q024'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4})

dadosE['Q025'] = dadosE['Q025'].map({'A': 0, 'B': 1})

#//////////////////

print(dadosE['NU_NOTA_MT'])

#df1 = np.nan_to_num(dadosE['NU_NOTA_MT'])

#dados = pd.DataFrame(df1)

#a = dadosE['NU_NOTA_MT']

#b = dados[0]

#dadosE['NU_NOTA_MT'] = dados.replace(a, b)

dadosE = dadosE.dropna(subset=['NU_NOTA_MT'])

print(dadosE['NU_NOTA_MT'])

soma = 0
media = 0

for i in range(1,100):
    x = dadosE[['TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_COR_RACA', 'TP_ESCOLA', 'TP_DEPENDENCIA_ADM_ESC', 'TP_LOCALIZACAO_ESC']]
    y = dadosE['NU_NOTA_MT']

    xtreino, xteste, ytreino, yteste = train_test_split(x, y, test_size= 0.30)

    modelo = LinearRegression()

    modelo.fit(xtreino, ytreino)

    preditos = modelo.predict(xteste)

    erro = mean_squared_error(yteste, preditos)

    soma = soma + erro

    media = soma / i

print('EQM da regressão: {}'.format(media,2))