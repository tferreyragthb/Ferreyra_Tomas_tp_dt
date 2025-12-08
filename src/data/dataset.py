import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, states, actions, returns_to_go, seq_len):
        self.states = states
        self.actions = actions
        self.returns_to_go = returns_to_go
        self.seq_len = seq_len

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # Recuperar trayectoria i
        states = self.states[idx]
        actions = self.actions[idx]
        rtg = self.returns_to_go[idx]

        # Truncamos o rellenamos a seq_len
        states = states[-self.seq_len:]
        actions = actions[-self.seq_len:]
        rtg = rtg[-self.seq_len:]

        # Convertimos a tensores si no lo son
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rtg = torch.tensor(rtg, dtype=torch.float32)

        # === SOLO DEVUELVE 3 COSAS ===
        return states, actions, rtg
