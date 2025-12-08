%%writefile /content/Ferreyra_Tomas_tp_dt/src/data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset


class RecommendationDataset(Dataset):
    """
    Dataset para entrenar el Decision Transformer.
    Usa ventanas de longitud context_length sobre las trayectorias.
    """

    def __init__(self, trajectories, context_length: int = 20):
        self.trajectories = trajectories
        self.context_length = context_length

        # Crear índices de (traj_idx, start_pos)
        self.indices = []
        for traj_idx, traj in enumerate(trajectories):
            T = len(traj["items"])
            for start in range(0, T):
                self.indices.append((traj_idx, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start = self.indices[idx]
        traj = self.trajectories[traj_idx]

        items = traj["items"]
        ratings = traj["ratings"]
        returns = traj["returns_to_go"]
        timesteps = traj["timesteps"]
        group = traj["user_group"]

        end = start + self.context_length

        seq_items = items[start:end]
        seq_ratings = ratings[start:end]
        seq_returns = returns[start:end]
        seq_times = timesteps[start:end]

        # Padding
        pad = self.context_length - len(seq_items)
        if pad > 0:
            seq_items = np.pad(seq_items, (0, pad))
            seq_ratings = np.pad(seq_ratings, (0, pad))
            seq_returns = np.pad(seq_returns, (0, pad))
            seq_times = np.pad(seq_times, (0, pad))

        return {
            "items": torch.tensor(seq_items, dtype=torch.long),
            "ratings": torch.tensor(seq_ratings, dtype=torch.float32),
            "returns_to_go": torch.tensor(seq_returns, dtype=torch.float32),
            "timesteps": torch.tensor(seq_times, dtype=torch.long),
            "group": torch.tensor(group, dtype=torch.long),
        }


class TestDataset(Dataset):
    """
    Dataset para evaluación.
    Cada entrada es la secuencia completa de items vistos por el usuario de test.
    Se toma solo el último contexto de tamaño context_length.
    """

    def __init__(self, test_users, num_items, context_length):
        self.test_users = test_users
        self.num_items = num_items
        self.context_length = context_length

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        user = self.test_users[idx]

        # Secuencia de items vistos
        items = np.array(user["items"], dtype=np.int64)
        group = int(user["group"])

        if len(items) >= self.context_length:
            seq = items[-self.context_length:]
        else:
            pad = self.context_length - len(items)
            seq = np.pad(items, (pad, 0), constant_values=0)

        return {
            "items": torch.tensor(seq, dtype=torch.long),
            "group": torch.tensor(group, dtype=torch.long),
        }
