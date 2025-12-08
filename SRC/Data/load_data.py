import os
import pandas as pd
import json


# ============================================================
# CONFIGURACIÓN DE PATHS
# ============================================================

def get_dataset_paths(dataset="netflix"):
    """
    Devuelve los paths correctos según el dataset elegido.
    No depende de __file__, funciona tanto local como en Colab.
    """
    base = os.path.join("data")  # Carpeta raíz donde está el dataset

    if dataset == "netflix":
        return {
            "train": os.path.join(base, "train", "netflix8_train.df"),
            "test": os.path.join(base, "test_users", "netflix8_test.json"),
            "groups": os.path.join(base, "groups", "mu_netflix8.csv"),
            "num_items": 752
        }

    elif dataset == "goodreads":
        return {
            "train": os.path.join(base, "train", "goodreads8_train.df"),
            "test": os.path.join(base, "test_users", "goodreads8_test.json"),
            "groups": os.path.join(base, "groups", "mu_goodreads8.csv"),
            "num_items": 472
        }

    else:
        raise ValueError("Dataset debe ser 'netflix' o 'goodreads'.")


# ============================================================
# FUNCIONES DE CARGA
# ============================================================

def load_train(dataset="netflix"):
    """
    Carga el DataFrame de entrenamiento.
    """
    paths = get_dataset_paths(dataset)
    return pd.read_pickle(paths["train"])


def load_test(dataset="netflix"):
    """
    Carga los usuarios de test (cold start).
    """
    paths = get_dataset_paths(dataset)
    with open(paths["test"], "r") as f:
        return json.load(f)


def load_group_centroids(dataset="netflix"):
    """
    Carga los centroides de cada grupo (opcional).
    """
    paths = get_dataset_paths(dataset)
    return pd.read_csv(paths["groups"], header=None)


# ============================================================
# MAIN (para pruebas locales)
# ============================================================

if __name__ == "__main__":
    df_train = load_train("netflix")
    test_users = load_test("netflix")
    mu = load_group_centroids("netflix")

    print("✓ Train cargado:", df_train.shape)
    print("✓ Test cargado:", len(test_users))
    print("✓ Centroides cargados:", mu.shape)
