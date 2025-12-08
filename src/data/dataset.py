import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    """
    Convierte trayectorias preprocesadas en tensores para Decision Transformer.
    """

    def __init__(self, trajectories, max_len=50):
        self.trajectories = trajectories
        self.max_len = max_len

    def __len__(self):
        return len(self.trajectories)

    def pad_or_truncate(self, seq):
        """
        Rellena con ceros o trunca secuencias al largo max_len.
        """
        seq = np.array(seq)
        if len(seq) >= self.max_len:
            return seq[-self.max_len:]
        else:
            pad_len = self.max_len - len(seq)
            return np.concatenate([np.zeros(pad_len), seq])

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        items = self.pad_or_truncate(traj["items"])
        ratings = self.pad_or_truncate(traj["ratings"])
        rtg = self.pad_or_truncate(traj["returns_to_go"])
        timesteps = self.pad_or_truncate(traj["timesteps"])

        group = traj["user_group"]

        return {
            "states": torch.tensor(items, dtype=torch.long),
            "actions": torch.tensor(ratings, dtype=torch.float32),
            "returns_to_go": torch.tensor(rtg, dtype=torch.float32
