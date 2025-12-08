import os
import pickle
import numpy as np
import torch

def normalize_item_ids(df_train):
    """
    Convierte los item_ids arbitrarios (ej: 472, 510, 338...)
    a IDs consecutivos desde 0 ... N-1
    """

    all_items = set()
    for seq in df_train["items"]:
        all_items.update(seq)

    all_items = sorted(all_items)
    mapping = {old: new for new, old in enumerate(all_items)}

    # aplicar el mapeo
    df_train["items"] = df_train["items"].apply(
        lambda seq: np.array([mapping[i] for i in seq])
    )

    return mapping, len(all_items)


def create_dt_dataset(df_train, save_dir="data/processed/", gamma=1.0):
    """
    Crea los 3 archivos necesarios para entrenar el Decision Transformer:
      - states.pkl
      - actions.pkl
      - returns_to_go.pkl

    A partir del dataframe crudo (items + ratings)
    """

    print("📦 Generando dataset procesado...")

    os.makedirs(save_dir, exist_ok=True)

    # Obtener listas crudas
    items = df_train["items"].tolist()
    ratings = df_train["ratings"].tolist()

    # ===============================
    # Crear secuencias DT
    # ===============================
    states = []
    actions = []
    rtg = []

    for item_seq, rating_seq in zip(items, ratings):

        n = len(item_seq)

        seq_states = []
        seq_actions = []
        seq_rtg = []

        for t in range(n):
            # Estado = todos los ítems vistos hasta t-1
            state_t = item_seq[:t]

            # Acción = ítem actual
            action_t = int(item_seq[t])

            # RTG = suma de ratings futuros
            rtg_t = float(sum(rating_seq[t:]))

            seq_states.append(state_t)
            seq_actions.append(action_t)
            seq_rtg.append(rtg_t)

        states.append(seq_states)
        actions.append(seq_actions)
        rtg.append(seq_rtg)

    # ===============================
    # Guardar archivos procesados
    # ===============================

    with open(os.path.join(save_dir, "states.pkl"), "wb") as f:
        pickle.dump(states, f)

    with open(os.path.join(save_dir, "actions.pkl"), "wb") as f:
        pickle.dump(actions, f)

    with open(os.path.join(save_dir, "returns_to_go.pkl"), "wb") as f:
        pickle.dump(rtg, f)

    print(f"✅ Dataset procesado guardado en {save_dir}")
    print("   - states.pkl")
    print("   - actions.pkl")
    print("   - returns_to_go.pkl")

    return states, actions, rtg
