import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    """
    Dataset usado para entrenamiento del Decision Transformer.
    Cada ítem del dataset es una ventana (context_length) sobre:
    - items
    - ratings (rewards)
    - returns_to_go
    """
    def __init__(self, trajectories, context_length):
        self.trajectories = trajectories
        self.context_length = context_length

        # Construir una lista de (trajectory_index, start_pos)
        self.indices = []
        for traj_idx, traj in enumerate(trajectories):
            T = len(traj["items"])
            for start in range(0, T):
                self.indices.append((traj_idx, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_pos = self.indices[idx]
        traj = self.trajectories[traj_idx]

        items = traj["items"]
        ratings = traj["ratings"]
        user_group = traj["user_group"]

        # Cortamos ventana
        end_pos = start_pos + self.context_length
        seq_items = items[start_pos:end_pos]
        seq_ratings = ratings[start_pos:end_pos]

        # Padding
        pad_len = self.context_length - len(seq_items)
        if pad_len > 0:
            seq_items = np.pad(seq_items, (0, pad_len), constant_values=0)
            seq_ratings = np.pad(seq_ratings, (0, pad_len), constant_values=0)

        return {
            "items": torch.tensor(seq_items, dtype=torch.long),
            "ratings": torch.tensor(seq_ratings, dtype=torch.float32),
            "group": torch.tensor(user_group, dtype=torch.long),
        }


class TestDataset(Dataset):
    """
    Dataset para evaluación.
    Cada entrada corresponde a un usuario de test y su secuencia completa de items vistos.
    Aquí sólo preparamos el input inicial para que el modelo recomiende el próximo ítem.
    """
    def __init__(self, test_users, num_items, context_length):
        self.test_users = test_users
        self.num_items = num_items
        self.context_length = context_length

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        user = self.test_users[idx]

        items = np.array(user["items"], dtype=np.int64)
        group = int(user["group"])

        # Tomar últimos context_length items como input al modelo
        if len(items) >= self.context_length:
            seq = items[-self.context_length:]
        else:
            pad = self.context_length - len(items)
            seq = np.pad(items, (pad, 0), constant_values=0)

        return {
            "items": torch.tensor(seq, dtype=torch.long),
            "group": torch.tensor(group, dtype=torch.long),
        }
