import os
import pickle
import numpy as np

def normalize_item_ids(df_train):
    """
    Normaliza los item_ids del dataset crudo a IDs consecutivos [0..N-1].
    Esto permite mapear tanto el train como el test al mismo espacio.
    """
    all_items = sorted({i for seq in df_train["items"] for i in seq})
    mapping = {old: new for new, old in enumerate(all_items)}

    df_train["items"] = df_train["items"].apply(
        lambda seq: np.array([mapping[i] for i in seq], dtype=np.int64)
    )

    return mapping, len(all_items)


def create_dt_dataset(df_train, save_path="data/processed/trajectories_train.pkl"):
    """
    Convierte el DataFrame crudo al formato Decision Transformer.

    Cada usuario produce UNA trayectoria:

        {
            'items': ndarray(int64),
            'ratings': ndarray(float32),
            'returns_to_go': ndarray(float32),
            'timesteps': ndarray(int64),
            'user_group': int
        }
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    trajectories = []

    for _, row in df_train.iterrows():
        items = row["items"].astype(np.int64)
        ratings = row["ratings"].astype(np.float32)
        group = int(row["user_group"])

        n = len(items)

        # === returns-to-go acumulado hacia adelante ===
        returns = np.zeros(n, dtype=np.float32)
        returns[-1] = ratings[-1]

        for t in range(n - 2, -1, -1):
            returns[t] = ratings[t] + returns[t + 1]

        timesteps = np.arange(n, dtype=np.int64)

        trajectory = {
            "items": items,
            "ratings": ratings,
            "returns_to_go": returns,
            "timesteps": timesteps,
            "user_group": group
        }

        trajectories.append(trajectory)

    with open(save_path, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"✅ Guardado dataset DT en {save_path}")
    print(f"   Total trayectorias: {len(trajectories)}")

    return trajectories
