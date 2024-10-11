import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge


def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

input_filepath='./data/processed'

fX_train = f"{input_filepath}/X_train_scaled.csv"
f_y = f"{input_filepath}/y_train.csv"

    # Import datasets
X_train = import_dataset(fX_train,header=None)
y_train = import_dataset(f_y)

# Modèle de régression à utiliser : Ridge Regression
ridge = Ridge()

# Définir les paramètres à tester dans la grille
param_grid = {
    'alpha': [0.1,0.5, 1.0,2.0,5.0,8.0, 10.0,15.0,50.0,75.0, 100.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}

# Initialiser la recherche par grille
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')

# Lancer la recherche des meilleurs paramètres
grid_search.fit(X_train, y_train)

# Récupérer les meilleurs paramètres et les meilleurs scores
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Meilleurs paramètres : {best_params}")
print(f"Meilleur score (MSE négatif) : {best_score}")

# Sauvegarder les param dans un fichier .pkl
model_filename = './models/best_param_ridge_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_params, file)