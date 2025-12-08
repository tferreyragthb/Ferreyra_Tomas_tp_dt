import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SequenceDataset(Dataset):
    def __init__(self, states, actions, returns_to_go, seq_len=50, num_items=752):
        self.seq_len = seq_len
        self.states = states
        self.actions = actions
        self.returns_to_go = returns_to_go
        self.num_items = num_items   # número total de ítems (netflix=752)

    def __len__(self):
        return len(self.states)

    def pad_or_truncate(self, seq, pad_value=0):
        seq = torch.tensor(seq, dtype=torch.long)

        if len(seq) >= self.seq_len:
            return seq[:self.seq_len]

        pad_size = self.seq_len - len(seq)
        pad = torch.full((pad_size,), pad_value, dtype=torch.long)
        return torch.cat([seq, pad], dim=0)

    def __getitem__(self, idx):
        # índices (T,)
        s_idx = self.pad_or_truncate(self.states[idx])
        a_idx = self.pad_or_truncate(self.actions[idx])
        rtg   = self.pad_or_truncate(self.returns_to_go[idx]).float()

        # one-hot (T, num_items)
        s = F.one_hot(s_idx, num_classes=self.num_items).float()
        a = F.one_hot(a_idx, num_classes=self.num_items).float()

        # RTG → (T, 1)
        rtg = rtg.unsqueeze(-1)

        return s, a, rtg
