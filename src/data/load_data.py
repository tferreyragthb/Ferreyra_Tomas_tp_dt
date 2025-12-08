import os
import pandas as pd
import json

# Directorio raíz del repo
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def _resolve_train_filename(dataset: str) -> str:
    """
    Devuelve el nombre correcto del archivo .df de train según el dataset.
    Los archivos que tenemos en data/train son:
      - netflix8_train.df
      - netflixM8_train.df
      - goodreads8_train.df
    """
    dataset = dataset.strip().lower()

    if dataset == "netflix":
        return "netflix8_train.df"
    elif dataset == "goodreads":
        return "goodreads8_train.df"
    elif dataset in ("netflixm", "netflixm8"):
        return "netflixM8_train.df"
    else:
        raise ValueError(f"Dataset no reconocido para train: {dataset}")


def _resolve_test_filename(dataset: str) -> str:
    """
    Archivos de test_users del repo de la materia:
      - netflix8_test.json
      - goodreads8_test.json
    """
    dataset = dataset.strip().lower()

    if dataset == "netflix":
        return "netflix8_test.json"
    elif dataset == "goodreads":
        return "goodreads8_test.json"
    else:
        # si en algún momento agregan otros, se puede extender
        raise ValueError(f"Dataset no reconocido para test: {dataset}")


def _resolve_groups_filename(dataset: str) -> str:
    """
    Archivos de centroides:
      - mu_netflix8.csv
      - mu_netflixM8.csv
      - mu_goodreads8.csv
    """
    dataset = dataset.strip().lower()

    if dataset == "netflix":
        return "mu_netflix8.csv"
    elif dataset in ("netflixm", "netflixm8"):
        return "mu_netflixM8.csv"
    elif dataset == "goodreads":
        return "mu_goodreads8.csv"
    else:
        raise ValueError(f"Dataset no reconocido para groups: {dataset}")


def load_train(dataset: str = "netflix") -> pd.DataFrame:
    """
    Carga el archivo .df de entrenamiento.
    """
    fname = _resolve_train_filename(dataset)
    path = os.path.join(DATA_DIR, "train", fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    return pd.read_pickle(path)


def load_test(dataset: str = "netflix"):
    """
    Carga el archivo JSON de test_users.
    """
    fname = _resolve_test_filename(dataset)
    path = os.path.join(DATA_DIR, "test_users", fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    with open(path, "r") as f:
        return json.load(f)


def load_group_centroids(dataset: str = "netflix") -> pd.DataFrame:
    """
    Carga la matriz de centroides (8 x num_items).
    """
    fname = _resolve_groups_filename(dataset)
    path = os.path.join(DATA_DIR, "groups", fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    return pd.read_csv(path, header=None)
