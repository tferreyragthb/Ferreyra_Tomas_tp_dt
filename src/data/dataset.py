import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, states, actions, returns_to_go, seq_len=50):
        """
        states: lista de arrays de tamaño variable
        actions: lista de arrays de tamaño variable
        returns_to_go: lista de arrays de tamaño variable
        """
        self.seq_len = seq_len
        self.states = states
        self.actions = actions
        self.returns_to_go = returns_to_go

    def __len__(self):
        return len(self.states)

    def pad_or_truncate(self, seq, pad_value=0):
        """Pad/trunca cada secuencia a (seq_len,)"""
        seq = torch.tensor(seq, dtype=torch.float32)

        if len(seq) >= self.seq_len:
            return seq[:self.seq_len]

        pad_size = self.seq_len - len(seq)
        pad = torch.full((pad_size,), pad_value)
        return torch.cat([seq, pad], dim=0)

    def __getitem__(self, idx):
        # (T,) → padificado a (seq_len,)
        s = self.pad_or_truncate(self.states[idx])
        a = self.pad_or_truncate(self.actions[idx])
        rtg = self.pad_or_truncate(self.returns_to_go[idx])

        # 🚀 Ahora agregamos la dimensión "state_dim"
        s = s.unsqueeze(-1)      # (seq_len, 1)
        a = a.unsqueeze(-1)      # (seq_len, 1)
        rtg = rtg.unsqueeze(-1)  # (seq_len, 1)

        return s, a, rtg
