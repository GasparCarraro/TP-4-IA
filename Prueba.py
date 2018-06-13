from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import scikitplot as skplt

df_data_train = pd.read_csv('train.csv', header=0)

accuracy = np.zeros(100)

# Borra las columnas no utilizadas para el entrenamiento
def delete_columns(df):
    return df.drop(['Name', 'PassengerId', 'Survived'], 1)


def delete_null_columns(df):
    return df.drop(['Cabin', 'Ticket'], 1)


def embarked_def(df):
    # Convert the Embarked training feature into dummies using one-hot
    # and leave one first category to prevent perfect collinearity
    train_embarked_dummied = pd.get_dummies(df["Embarked"], prefix='Embarked', drop_first=True)

    # Concatenate the dataframe of dummies with the main dataframes
    df = pd.concat([df.drop('Embarked', axis=1), train_embarked_dummied], axis=1)

    return df


def sex_def(df):
    # Create an encoder
    sex_encoder = preprocessing.LabelEncoder()

    # Fit the encoder to the train data so it knows that male = 1
    sex_encoder.fit(df_data_train['Sex'])

    # Apply the encoder to the training data
    df['Sex'] = sex_encoder.transform(df['Sex'])

    return df


def age_def(df):
    # Create an imputer object
    age_imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)

    # Fit the imputer object on the training data
    age_imputer.fit(df['Age'].values.reshape(-1, 1))

    # Apply the imputer object to the training and test data
    df['Age'] = age_imputer.transform(df['Age'].values.reshape(-1, 1))

    return df


def pclass_def(df):
    # Convert the Pclass training feature into dummies using one-hot
    # and leave one first category to prevent perfect collinearity
    train_Pclass_dummied = pd.get_dummies(df["Pclass"], prefix='Pclass', drop_first=True)

    # Concatenate the dataframe of dummies with the main dataframes
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

X_train = df_X_train.values
y_train = df_y_train.values


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=10, test_size=0.3)

n_estimators = range(1, 5001, 1000)
criterion = ['gini','entropy']
max_features = range(1, 9, 2)
max_depth = [None] + list(range(5, 25, 1))

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'max_features': max_features,
               'criterion': criterion}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = GridSearchCV(rf, random_grid)
# Fit the random search model
rf_random.fit(X_train, y_train.ravel())

print(rf_random.best_params_)
print(rf_random.best_score_)
