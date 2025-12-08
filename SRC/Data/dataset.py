import torch
from torch.utils.data import Dataset

class DTDataset(Dataset):
    """
    Dataset para Decision Transformer.
    Recibe una lista de trayectorias procesadas.
    """

    def __init__(self, trajectories, max_len=50):
        self.trajs = trajectories
        self.max_len = max_len

    def __len__(self):
        return len(self.trajs)

    def pad_sequence(self, seq):
        """
        Recorta o paddea las secuencias a max_len.
        """
        L = len(seq)

        if L >= self.max_len:
            return seq[:self.max_len]

        padded = torch.zeros(self.max_len, dtype=torch.long if seq.dtype == int else torch.float32)
        padded[:L] = torch.tensor(seq)
        return padded

    def __getitem__(self, idx):
        traj = self.trajs[idx]

        items = self.pad_sequence(traj["items"])
        ratings = self.pad_sequence(traj["ratings"])
        rtg = self.pad_sequence(traj["returns_to_go"])
        timesteps = self.pad_sequence(traj["timesteps"])
        group = torch.tensor(traj["user_group"], dtype=torch.long)

        return {
            "items": items,
            "ratings": ratings,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "user_group": group,
        }
