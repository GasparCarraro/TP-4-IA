from sklearn.model_selection import RandomizedSearchCV
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

# Number of trees in random forest
n_estimators = range(1, 100)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = range(1, 10)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_state = range(100, 500, 100)
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'random_state': random_state}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(rf, random_grid, cv=10)
# Fit the random search model
rf_random.fit(X_train, y_train.ravel())

y_hat = rf_random.predict(X_test)

print(accuracy_score(y_true=y_test, y_pred=y_hat))

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(rf_random.best_params_))
print("Best score is {}".format(rf_random.best_score_))
