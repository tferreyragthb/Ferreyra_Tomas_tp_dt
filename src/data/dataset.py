import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, states, actions, returns_to_go, seq_len=50):
        self.states = states
        self.actions = actions
        self.returns_to_go = returns_to_go
        self.seq_len = seq_len

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        s = self.states[idx]
        a = self.actions[idx]
        rtg = self.returns_to_go[idx]

        # Convert to tensors
        s = torch.tensor(s, dtype=torch.long)
        a = torch.tensor(a, dtype=torch.float)
        rtg = torch.tensor(rtg, dtype=torch.float)

        # recortar al final si es más largo
        if len(s) > self.seq_len:
            s = s[-self.seq_len:]
            a = a[-self.seq_len:]
            rtg = rtg[-self.seq_len:]

        # rellenar con ceros si es más corto
        if len(s) < self.seq_len:
            pad = self.seq_len - len(s)

            s = torch.cat([torch.zeros(pad, dtype=torch.long), s])
            a = torch.cat([torch.zeros(pad, dtype=torch.float), a])
            rtg = torch.cat([torch.zeros(pad, dtype=torch.float), rtg])

        return s, a, rtg
