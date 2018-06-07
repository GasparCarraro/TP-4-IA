import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt

df_data_train = pd.read_csv('train.csv', header=0)
#df_data_test = pd.read_csv('test.csv', header=0)

accuracy = np.zeros(100)

# Cantidad total de filas
# print("Cantidad total de filas TRAIN %s" % len(df_data_train))
#print("Cantidad total de filas TEST %s" % len(df_data_test))
# Contamos cantidad de datos null por cada columna
# print(len(df_data_train) - df_data_train.count())
#print(len(df_data_test) - df_data_test.count())


# Borra las columnas no utilizadas para el entrenamiento
def delete_columns(df):
    return df.drop(['Name', 'PassengerId', 'Survived'], 1)


def delete_null_columns(df):
    return df.drop(['Cabin', 'Ticket'], 1)


def embarked_def(df):
    embarked = {'C': 1, 'Q': 2, 'S': 3}
    df.Embarked = df.Embarked.fillna(0)
    df.Embarked = df.Embarked .replace(embarked)
    return df


def sex_def(df):
    sex = {'male': 1, 'female': 2}
    df.Sex = df.Sex.fillna(0)
    df.Sex = df.Sex.replace(sex)
    return df


def age_def(df):
    df.Age = df.Age.fillna(-1)
    return df


df_X_train = df_data_train
#df_X_test = df_data_test
df_X_train = delete_columns(df_X_train)
#df_X_test = delete_columns(df_X_test)
df_X_train = delete_null_columns(df_X_train)
#df_X_test = delete_null_columns(df_X_test)
df_y_train = df_data_train[['Survived']]
#df_y_test = df_data_test[['Survived']]

# print(df_X_train)
# print(df_y_train)

df_X_train = embarked_def(df_X_train)
#df_X_test = embarked_def(df_X_test)
df_X_train = sex_def(df_X_train)
#df_X_test = sex_def(df_X_test)
df_X_train = age_def(df_X_train)
#df_X_test = age_def(df_X_test)

X_train = df_X_train.values
y_train = df_y_train.values


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=10, test_size=0.3)

# ------------------------------------Sin preprocesamiento--------------------------------------
# Hacer un For para que vaya tomando distintos valores en max_depth.

mejor_k = 0
mejor_tasa_aciertos = 0

for k in range(1, 101):
    clf = RandomForestClassifier(n_estimators=k)
    clf.fit(X_train, y_train.ravel())

    y_hat = clf.predict(X_test)

    tasa_aciertos = accuracy_score(y_true=y_test, y_pred=y_hat)

    accuracy[k-1] = tasa_aciertos

    if tasa_aciertos > mejor_tasa_aciertos:
        mejor_tasa_aciertos = tasa_aciertos
        mejor_k = k

clf = RandomForestClassifier(n_estimators=mejor_k)
clf.fit(X_train, y_train.ravel())

y_hat = clf.predict(X_test)

print("---------Sin preprocesamiento----------")
print("Aciertos para %s arboles: " % mejor_k, mejor_tasa_aciertos)

print(metrics.classification_report(y_test, y_hat))

# -------------------------------------Normalizacion--------------------------------------
mean_X_train = np.mean(X_train)
stdv_X_train = np.std(X_train)
X_train_scaled = (X_train - mean_X_train) / stdv_X_train
X_test_scaled = (X_test - mean_X_train) / stdv_X_train

mejor_k = 0
mejor_tasa_aciertos = 0

for k in range(1, 101):
    clf = RandomForestClassifier(n_estimators=k)
    clf.fit(X_train_scaled, y_train.ravel())

    y_hat = clf.predict(X_test_scaled)

    tasa_aciertos = accuracy_score(y_true=y_test, y_pred=y_hat)

    accuracy[k-1] = tasa_aciertos

    if tasa_aciertos > mejor_tasa_aciertos:
        mejor_tasa_aciertos = tasa_aciertos
        mejor_k = k

# Hacer un For para que vaya tomando distintos valores en max_depth.
clf = RandomForestClassifier(n_estimators=mejor_k)
clf.fit(X_train_scaled, y_train.ravel())

y_hat = clf.predict(X_test_scaled)

print("---------Con normalizacion manual----------")
print("Aciertos para %s arboles: " % mejor_k, mejor_tasa_aciertos)
print(metrics.classification_report(y_test, y_hat))

# plt.cla()
plt.bar(range(100), accuracy)
plt.show()
