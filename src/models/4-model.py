import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

input_filepath='./data/processed'

fX_train = f"{input_filepath}/X_train_scaled.csv"
f_ytrain = f"{input_filepath}/y_train.csv"
fX_test = f"{input_filepath}/X_test_scaled.csv"
f_ytest = f"{input_filepath}/y_test.csv"
    # Import datasets
X_train = import_dataset(fX_train,header=None)
y_train = import_dataset(f_ytrain)

# import best param du fichier .pkl
model_filename = 'models/best_param_ridge_model.pkl'
with open(model_filename, 'rb') as file:
    best_params=pickle.load(file)


# Créer le modèle Ridge avec les meilleurs paramètres
model = Ridge(alpha=best_params['alpha'], solver=best_params['solver'])

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Sauvegarder le modèle entraîné dans un fichier .pkl
model_filename = 'models/trained_ridge_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Modèle entraîné sauvegardé sous {model_filename}")