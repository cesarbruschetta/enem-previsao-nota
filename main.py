# Importar as bibliotecas a serem usadas
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Leitura dos datasets de treino e teste e criação do df de resposta
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_resposta = pd.DataFrame()

# Verificar se os dados de teste estão nos dados de treinamento
print(set(df_test.columns).issubset(set(df_train.columns)))

# Salvar os dados das inscrições
df_resposta["NU_INSCRICAO"] = df_test["NU_INSCRICAO"]


# Selecionar somente valores inteiros e floats
df_test = df_test.select_dtypes(include=["int64", "float64"])

# var = [
#     "NU_IDADE",
#     "IN_TREINEIRO",
#     "NU_NOTA_CN", #- ciências da natureza: 2
#     "NU_NOTA_CH", #- ciências humanas: 1
#     "NU_NOTA_LC", #- linguagens e códigos: 1.5
#     "NU_NOTA_REDACAO", #- redação: 3
# ]
# df_test[var].corr()

# Usando o loc com uma condição composta para obter somente registros com todas as provas
df_train = df_train.loc[
    (df_train["NU_NOTA_CN"].notnull())
    & (df_train["NU_NOTA_CH"].notnull())
    & (df_train["NU_NOTA_LC"].notnull())
    & (df_train["NU_NOTA_REDACAO"].notnull())
    & (df_train["NU_NOTA_MT"].notnull())
]


# Preencher valores nulos com o valor médio - Tratamento das notas de provas corrompidas
df_train["NU_NOTA_CN"].fillna(df_train["NU_NOTA_CN"].mean(), inplace=True)
df_train["NU_NOTA_CH"].fillna(df_train["NU_NOTA_CH"].mean(), inplace=True)
df_train["NU_NOTA_REDACAO"].fillna(df_train["NU_NOTA_REDACAO"].mean(), inplace=True)
df_train["NU_NOTA_LC"].fillna(df_train["NU_NOTA_LC"].mean(), inplace=True)

df_test["NU_NOTA_CN"].fillna(df_train["NU_NOTA_CN"].mean(), inplace=True)
df_test["NU_NOTA_CH"].fillna(df_train["NU_NOTA_CH"].mean(), inplace=True)
df_test["NU_NOTA_REDACAO"].fillna(df_train["NU_NOTA_REDACAO"].mean(), inplace=True)
df_test["NU_NOTA_LC"].fillna(df_train["NU_NOTA_LC"].mean(), inplace=True)


y = df_train["NU_NOTA_MT"]

# Definição do dataset de treino somente com as informações relevantes para treinar o modelo
features = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_REDACAO"]
x_train = df_train[features]

# Ajustando o Transformer API
scaler = preprocessing.StandardScaler().fit(x_train)


X_train_scaled = scaler.transform(x_train)
print('Media: {}'.format(X_train_scaled.mean(axis=1)))
print('Desvio Padrao: {}'.format(X_train_scaled.std(axis=0)))


x_test = df_test[features]

# n_estimators=100 (número de nós) ,
# n_jobs=-1 ( todo o processamento possível) ,
# warm_start=True (mantém o aprendizado e reprocessa o modelo, melhorando-o)
pipeline = make_pipeline(
    preprocessing.StandardScaler(),
    RandomForestRegressor(n_estimators=200, n_jobs=-1, warm_start=True),
)


# max_features : O número de features a considerar quando pesquisar pela melhor separação (testará as 3 opções e identificará a melhor para o modelo)
# max_depth :  Profundidade máxima da árvore de decisão. Se None (nenhuma), os nós serão expandidos até acabar as folhas ou até que elas contenham o mínimo valor de amostras possível.
hyperparameters = {
    "randomforestregressor__max_features": ["auto", "sqrt", "log2"],
    "randomforestregressor__max_depth": [None, 5, 3, 1],
}


# Ajustar e sintonizar o modelo
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train_scaled, y)


pred_notas = clf.predict(x_test)
df_resposta['NU_NOTA_MT'] = np.around(pred_notas,2)


df_resposta.to_csv("answer.csv", index=False, header=True)


# Salvar o modelo preditivo
joblib.dump(clf, "rf_regressor.pkl")

# Usar/carregar o modelo preditivo
clf2 = joblib.load("rf_regressor.pkl")
clf2.predict(x_test)
