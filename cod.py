#importar las dependencias
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

#creamos un dataframe
df= pd.read_csv('data1.csv', sep=';')
df.head()
df.info()

#cambiamos location code a categorico
df['location code'].unique()
df['location code'] = df['location code'].astype(str)

#cambiamos el tipo de dato de  credit card info save y push status
df['credit card info save'] = df['credit card info save'].replace({'yes': 1, 'no':0})
df['push status'] = df['push status'].replace({'yes': 1, 'no':0})
#cambiamos coma por punto y tipo de dato
df['avg order value'] = df['avg order value'].str.replace(',', '.').astype(float)
df['discount rate per visited products'] = df['discount rate per visited products'].str.replace(',', '.').astype(float)
df['product detail view per app session'] = df['product detail view per app session'].str.replace(',', '.').astype(float)
df['add to cart per session'] = df['add to cart per session'].str.replace(',', '.').astype(float)

#codificacion de variable categorica
df = pd.get_dummies(df, columns=['location code'])

#eliminamos user id
df= df.drop('user id', axis= 1)

df.head()
df.columns

#escalamos los datos
cols_to_scale = ['account length',
       'add to wishlist', 'desktop sessions', 'app sessions',
       'desktop transactions', 'total product detail views',
       'session duration', 'promotion clicks', 'avg order value',
       'sale product views', 'discount rate per visited products',
       'product detail view per app session', 'app transactions',
       'add to cart per session', 'customer service calls']
scaler = Normalizer()
scaled_data =scaler.fit_transform(df[cols_to_scale])
scaled_df= pd.DataFrame(scaled_data, index=df.index, columns= cols_to_scale)

scaled_df.head()

#eliminamos columnas para evitar duplicas
df = df.drop(cols_to_scale, axis= 1)
df = pd.merge(df, scaled_df, left_index= True, right_index= True)

from numpy.random.mtrand import random
#dividimos los datos de entrenamiento
x= df.drop('churn', axis= 1)
y= df['churn']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.33, random_state= 42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#construimos modelo
xgb_cl = xgb.XGBClassifier()
xgb_cl.fit(x_train, y_train)

preds= xgb_cl.predict(x_test)
#medimos la precision
acc = accuracy_score(y_test, preds)
print('Precision del modelo de:', acc)

#mejoramiento de los hiperparametros
param_grid = {
    'max_depth':[5],
    'learning_rate':[0, 0.01, 0.05, 0.1],
    'gamma':[1, 5, 10],
    'scale_pos_weight': [2, 5, 10, 20],
    'subsample': [1],
    'colsample_bytree': [1]
}
xgb_c12= xgb.XGBClassifier(objective= 'binary:logistic')
grid_cv= GridSearchCV(xgb_c12, param_grid, n_jobs=-1, cv=3, scoring= 'roc_auc')
_ = grid_cv.fit(x_train, y_train)
print('El mejor puntaje:', grid_cv.best_score_)
print('El mejor parametro:', grid_cv.best_params_)

from xgboost.core import Objective
final_cl= xgb.XGBClassifier(
    **grid_cv.best_params_, Objective='binary:logistic'
)
grid_final= final_cl.fit(x_train, y_train)
preds= grid_final.predict(x_test)
acc= accuracy_score(y_test, preds)
print('El puntaje del modelo es:', acc)
