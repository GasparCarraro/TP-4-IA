import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

df_data_train = pd.read_csv('train.csv', header=0)
#df_data_test = pd.read_csv('test.csv', header=0)


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
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(X_train, y_train.ravel())

y_hat = clf.predict(X_test)

print("---------Sin preprocesamiento----------")
print("Aciertos: ", accuracy_score(y_true=y_test, y_pred=y_hat))
print(metrics.classification_report(y_test, y_hat))

# --------------------------------------Normalizacion--------------------------------------------
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hacer un For para que vaya tomando distintos valores en max_depth.
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(X_train_scaled, y_train.ravel())

y_hat = clf.predict(X_test_scaled)

print("---------Con normalizacion----------")
print("Aciertos: ", accuracy_score(y_true=y_test, y_pred=y_hat))
print(metrics.classification_report(y_test, y_hat))
