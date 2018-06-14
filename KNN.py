import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.decomposition import PCA
import scikitplot as skplt
from sklearn import preprocessing

# Importamos el dataset
df_data_train = pd.read_csv('train.csv', header=0)

# Borra las columnas no utilizadas para el entrenamiento
def delete_columns(df):
    return df.drop(['Name', 'PassengerId', 'Survived'], 1)

def delete_null_columns(df):
    return df.drop(['Cabin', 'Ticket'], 1)


def embarked_def(df):
    # Crea una columna por cada clase.
    train_embarked_dummied = pd.get_dummies(df["Embarked"], prefix='Embarked', drop_first=True)
    df = pd.concat([df.drop('Embarked', axis=1), train_embarked_dummied], axis=1)
    return df


def sex_def(df):
    sex_encoder = preprocessing.LabelEncoder()
    # Le asigno un valor numérico a cada clase
    sex_encoder.fit(df_data_train['Sex'])
    # Se aplica al entrenamiento
    df['Sex'] = sex_encoder.transform(df['Sex'])
    return df


def age_def(df):
    age_imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    # Fit the imputer object on the training data
    age_imputer.fit(df['Age'].values.reshape(-1, 1))
    # Apply the imputer object to the training and test data
    df['Age'] = age_imputer.transform(df['Age'].values.reshape(-1, 1))
    return df


def pclass_def(df):
    train_Pclass_dummied = pd.get_dummies(df["Pclass"], prefix='Pclass', drop_first=True)
    df = pd.concat([df, train_Pclass_dummied], axis=1)
    return df


def fare_def(df):
    # Create an imputer object
    fare_imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    # Fit the imputer object on the training data
    fare_imputer.fit(df['Fare'].values.reshape(-1, 1))
    # Apply the imputer object to the training and test data
    df['Fare'] = fare_imputer.transform(df['Fare'].values.reshape(-1, 1))
    return df



df_X_train = df_data_train
df_X_train = delete_columns(df_X_train)
df_X_train = delete_null_columns(df_X_train)
df_y_train = df_data_train[['Survived']]

df_X_train = embarked_def(df_X_train)
df_X_train = sex_def(df_X_train)
df_X_train = age_def(df_X_train)
df_X_train = pclass_def(df_X_train)
df_X_train = fare_def(df_X_train)


X_train = df_X_train.values
y_train = df_y_train.values

# Divido el dataset en entrenamiento y test.
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=10, test_size=0.3)

# Divido el entrenamiento, en entrenamiento y validación.
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

# Creo PCA con 4 componentes.
pca = PCA(n_components=4)

# Utilizo PCA con conjunto de entrenamiento normalizado.
pca.fit(normalizacionEntrenamiento)

X_transformedEntrenamiento = pca.transform(normalizacionEntrenamiento)

X_transformedValidacion = pca.transform(normalizacionValidacion)

X_transformedTest = pca.transform(normalizacionTest)

# Inicializo variables que vamos a utilizar.
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

# Graficar PCA.
y_pos = np.arange(4)
# pca.explained_variance_ratio_ es quien nos devuelve el gráfico de la varianza
plt.bar(y_pos, np.round(100 * pca.explained_variance_ratio_,
                        decimals=1), align='center', alpha=0.5)
plt.xticks(y_pos, [1,2,3,4])
plt.xlabel('N° de Componente Principal')
plt.ylabel('Varianza')
plt.title('Varianza explicada por cada componente')
plt.show()

# Obtener métricas curva ROC.
y_hat_probas = clasificadorNN.predict_proba(X_transformedTest)

# Gráfico de curva ROC.
skplt.metrics.plot_roc_curve(y_test, y_hat_probas)
plt.show()

# CONCLUSIONES:
# Probamos PCA con distinta cantidad de componentes y la mejor fue 4.
# Convenia normalizar los datos.
# Para obtener mejores hiperparametros, dividimos el conjunto en validacion (15% entrenamiento).
