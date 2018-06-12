from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import scikitplot as skplt



df_data_train = pd.read_csv('train.csv', header=0)
#df_data_test = pd.read_csv('test.csv', header=0)

accuracy = np.zeros(100)

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

X_train2, X_validation, y_train2, y_validation = train_test_split(X_train, y_train, random_state=10, test_size=0.15)

# Calculo la media con una función de numpy.
media = np.mean(X_train2,0)

# Calculo la desviacion con una función de numpy.
desviacionEstandar = np.std(X_train2,0)

# Normalizo los datos de train.
normalizacionEntrenamiento = (X_train2 - media) / desviacionEstandar

# Normalizo los datos de test.
normalizacionTest = (X_test - media) / desviacionEstandar

# Normalizo los datos de validacion.
normalizacionValidacion = (X_validation - media) / desviacionEstandar

pca = PCA(n_components=4)

pca.fit(normalizacionEntrenamiento)
X_transformedEntrenamiento = pca.transform(normalizacionEntrenamiento)

X_transformedValidacion = pca.transform(normalizacionValidacion)

X_transformedTest = pca.transform(normalizacionTest)

mejorTasaDeAcierto = 0
actualTasaDeAcierto = 0
mejorK = 0

# Recorro 20 veces para buscar el mejor K.
for k in range(1, 21):
    clasificadorNN = neighbors.KNeighborsClassifier(n_neighbors=k)
    clasificadorNN.fit(X_transformedEntrenamiento, y_train2.ravel())
    y_hat = clasificadorNN.predict(X_transformedValidacion)
    actualTasaDeAcierto = accuracy_score(y_true=y_validation, y_pred=y_hat)

    # Comparo la tasa de acierto actual con la anterior.
    if actualTasaDeAcierto > mejorTasaDeAcierto:
        mejorTasaDeAcierto = actualTasaDeAcierto
        mejorK = k

print("La mayor tasa de acierto es", mejorTasaDeAcierto, "en el K =", mejorK)

# Instanciamos el clasificador para el mejor K.
clasificadorNN = neighbors.KNeighborsClassifier(n_neighbors=mejorK)
clasificadorNN.fit(X_transformedEntrenamiento, y_train2.ravel())
y_hat = clasificadorNN.predict(X_transformedTest)
tasaDeAcierto = accuracy_score(y_true=y_test, y_pred=y_hat)

print("Tasa de acierto del Test:", tasaDeAcierto)

# Muestro reporte de clasificación.
print(metrics.classification_report(y_test, y_hat))

y_pos = np.arange(4)
# pca.explained_variance_ratio_ es quien nos devuelve el gráfico de la varianza
plt.bar(y_pos, np.round(100 * pca.explained_variance_ratio_,
                        decimals=1), align='center', alpha=0.5)
plt.xticks(y_pos, [1,2,3,4])
plt.xlabel('N° de Componente Principal')
plt.ylabel('Varianza')
plt.title('Varianza explicada por cada componente')
plt.show()

y_hat_probas = clasificadorNN.predict_proba(X_transformedTest)

skplt.metrics.plot_roc_curve(y_test, y_hat_probas)
plt.show()
