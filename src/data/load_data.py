import os
import pandas as pd
import json

# Directorio raíz del repo
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def _resolve_train_filename(dataset):
    """
    Devuelve el nombre correcto del archivo .df según el dataset.
    """
    if dataset == "netflix":
        return "netflix_train.df"
    elif dataset == "goodreads":
        return "goodreads8_train.df"
    elif dataset == "netflixM":
        return "netflixM8_train.df"
    else:
        raise ValueError(f"Dataset no reconocido: {dataset}")


def _resolve_test_filename(dataset):
    """
    Devuelve el nombre correcto del archivo test_users JSON.
    """
    return f"{dataset}8_test.json"


def _resolve_groups_filename(dataset):
    """
    Devuelve archivo correcto de centroides.
    """
    return f"mu_{dataset}8.csv"


def load_train(dataset="netflix"):
    fname = _resolve_train_filename(dataset)
    path = os.path.join(DATA_DIR, "train", fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    return pd.read_pickle(path)


def load_test(dataset="netflix"):
    fname = _resolve_test_filename(dataset)
    path = os.path.join(DATA_DIR, "test_users", fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    with open(path, "r") as f:
        return json.load(f)


def load_group_centroids(dataset="netflix"):
    fname = _resolve_groups_filename(dataset)
    path = os.path.join(DATA_DIR, "groups", fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    return pd.read_csv(path, header=None)
