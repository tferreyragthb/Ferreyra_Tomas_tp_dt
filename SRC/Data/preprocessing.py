import numpy as np

def create_dt_dataset(df_train):
    """
    Convierte el dataset raw a formato compatible con Decision Transformer.
    
    Cada usuario se convierte en una trayectoria con:
      - items
      - ratings
      - returns_to_go
      - timesteps
      - user_group
    """
    
    trajectories = []

    for _, row in df_train.iterrows():
        items = np.array(row["items"], dtype=np.int32)
        ratings = np.array(row["ratings"], dtype=np.float32)
        group = int(row["user_group"])

        # returns-to-go (suma hacia adelante)
        rtg = np.zeros_like(ratings)
        rtg[-1] = ratings[-1]
        for t in range(len(ratings) - 2, -1, -1):
            rtg[t] = ratings[t] + rtg[t + 1]

        traj = {
            "items": items,
            "ratings": ratings,
            "returns_to_go": rtg,
            "timesteps": np.arange(len(items), dtype=np.int32),
            "user_group": group,
        }

        trajectories.append(traj)

    return trajectories
