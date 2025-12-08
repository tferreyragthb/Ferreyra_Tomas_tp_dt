import os
import pandas as pd
import json

# Directorio raíz del repo
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATA_DIR = os.path.join(REPO_ROOT, "data")

def load_train(dataset="netflix"):
    """
    Carga el archivo .df de entrenamiento.
    """
    path = os.path.join(DATA_DIR, "train", f"{dataset}8_train.df")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    return pd.read_pickle(path)

def load_test(dataset="netflix"):
    """
    Carga el archivo JSON de test_users.
    """
    path = os.path.join(DATA_DIR, "test_users", f"{dataset}8_test.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    with open(path, "r") as f:
        return json.load(f)

def load_group_centroids(dataset="netflix"):
    """
    Carga la matriz de centroides (8 x num_items).
    """
    path = os.path.join(DATA_DIR, "groups", f"mu_{dataset}8.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    return pd.read_csv(path, header=None)
