import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, df, max_len=50):
        self.df = df
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        items = row["items"]
        ratings = row["ratings"]

        T = min(len(items), self.max_len)

        # Dimensiones del item_id (máximo item + 1)
        n_items = max(max(items), 799) + 1

        states = torch.zeros(self.max_len, n_items)
        actions = torch.zeros(self.max_len, n_items)
        rtg = torch.zeros(self.max_len, 1)

        # Construcción de secuencias
        for i in range(T):
            item = items[i]
            rating = ratings[i]

            states[i, item] = 1.0
            actions[i, item] = rating / 5.0
            rtg[i] = sum(ratings[i:]) / 50.0

        return states, actions, rtg
