import os
import json
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

# Chargement des données pour évaluation (Remplacer par votre méthode de chargement)
input_filepath='./data/processed'
fX_train = f"{input_filepath}/X_train_scaled.csv"
f_ytrain = f"{input_filepath}/y_train.csv"
fX_test = f"{input_filepath}/X_test_scaled.csv"
f_ytest = f"{input_filepath}/y_test.csv"
    # Import datasets
X_train = import_dataset(fX_train,header=None)
y_train = import_dataset(f_ytrain)
X_test = import_dataset(fX_test,header=None)
y_test = import_dataset(f_ytest)

# Chemin vers le modèle sauvegardé
model_filename = 'models/trained_ridge_model.pkl'

# Charger le modèle entraîné
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

print("Modèle chargé avec succès !")

# Faire des prédictions avec le modèle chargé
y_pred = loaded_model.predict(X_test)

# Évaluation des performances
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Afficher les métriques
print(f"MSE : {mse}")
print(f"R² : {r2}")

# Sauvegarder les scores dans un fichier JSON dans le dossier 'metrics'
scores = {
    'Mean Squared Error': mse,
    'R2 Score': r2
}

scores_filename = 'metrics/scores.json'
with open(scores_filename, 'w') as f:
    json.dump(scores, f)

print(f"Métriques sauvegardées dans {scores_filename}")

# Sauvegarder les prédictions et les vraies valeurs dans un nouveau fichier CSV
predictions_df =y_test
predictions_df["y_pred"]=y_pred
data_filename = 'data/predictions.csv'
predictions_df.to_csv(data_filename, index=False)

print(f"Prédictions sauvegardées dans {data_filename}")
