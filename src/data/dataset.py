import numpy as np
import torch
from torch.utils.data import Dataset


class RecommendationDataset(Dataset):
    """
    Dataset para entrenar el Decision Transformer.

    Cada elemento es una ventana de longitud `context_length`
    tomada de una trayectoria producida por `create_dt_dataset`.
    """

    def __init__(self, trajectories, context_length: int = 20):
        """
        Args:
            trajectories: lista de dicts con keys:
                - 'items'
                - 'ratings'
                - 'returns_to_go'
                - 'timesteps'
                - 'user_group'
            context_length: largo de la ventana de contexto.
        """
        self.trajectories = trajectories
        self.context_length = context_length

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        # Secuencia completa
        items = traj["items"]
        rtg = traj["returns_to_go"]
        timesteps = traj["timesteps"]
        group = traj["user_group"]

        # Longitud efectiva de la secuencia
        seq_len = min(len(items), self.context_length)

        # Random start para data augmentation
        if len(items) > self.context_length:
            start_idx = np.random.randint(
                0, len(items) - self.context_length + 1
            )
        else:
            start_idx = 0

        end_idx = start_idx + seq_len

        # States / actions: usamos la misma secuencia de items
        states = items[start_idx:end_idx]
        actions = items[start_idx:end_idx]

        # Targets: próximo item (shifted)
        targets = np.zeros(seq_len, dtype=np.int64)
        if seq_len > 1:
            targets[:-1] = items[start_idx + 1 : end_idx]
        targets[-1] = -1  # padding para el último timestep

        # Returns-to-go (como columna)
        rtg_seq = rtg[start_idx:end_idx].reshape(-1, 1)

        # Timesteps
        time_seq = timesteps[start_idx:end_idx]

        return {
            "states": torch.tensor(states, dtype=torch.long),
            "actions": torch.tensor(actions, dtype=torch.long),
            "rtg": torch.tensor(rtg_seq, dtype=torch.float32),
            "timesteps": torch.tensor(time_seq, dtype=torch.long),
            "groups": torch.tensor(group, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
        }
