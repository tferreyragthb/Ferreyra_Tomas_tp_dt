import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, states, actions, returns_to_go, seq_len=50):
        self.seq_len = seq_len
        self.states = states
        self.actions = actions
        self.returns_to_go = returns_to_go

    def __len__(self):
        return len(self.states)

    def pad_or_truncate(self, seq, pad_value=0):
        seq = torch.tensor(seq, dtype=torch.long)   # 🔥 ENTEROS, NO float32

        if len(seq) >= self.seq_len:
            return seq[:self.seq_len]

        pad_size = self.seq_len - len(seq)
        pad = torch.full((pad_size,), pad_value, dtype=torch.long)
        return torch.cat([seq, pad], dim=0)

    def __getitem__(self, idx):
        s = self.pad_or_truncate(self.states[idx])         # (T,)
        a = self.pad_or_truncate(self.actions[idx])        # (T,)
        rtg = self.pad_or_truncate(self.returns_to_go[idx])# (T,)

        return s, a, rtg
