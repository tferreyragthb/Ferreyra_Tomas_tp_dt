import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, train_dataset, lr=1e-4, batch_size=8, device="cuda"):
        self.device = device
        self.model = model.to(device)
        self.loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.opt = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train(self, epochs=2):
        self.model.train()

        for ep in range(epochs):
            total_loss = 0

            for states, actions, rtg in self.loader:

                # ---- TIPOS CORRECTOS ---- #
                states = states.float().to(self.device)     # (B, T, state_dim)
                actions = actions.float().to(self.device)   # (B, T, action_dim)

                # aseguramos dimensión del RTG → (B, T, 1)
                if rtg.dim() == 2:
                    rtg = rtg.unsqueeze(-1)

                rtg = rtg.float().to(self.device)

                # ---- FORWARD ---- #
                pred = self.model(states, actions, rtg)

                # Queremos predecir el action siguiente (teacher forcing)
                # pred[:, :-1, :] vs true_actions[:, 1:, :]
                loss = self.loss_fn(
                    pred[:, :-1, :],
                    actions[:, 1:, :]
                )

                # ---- BACKPROP ---- #
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total_loss += loss.item()

            print(f"Epoch {ep+1}/{epochs} - Loss: {total_loss:.4f}")
