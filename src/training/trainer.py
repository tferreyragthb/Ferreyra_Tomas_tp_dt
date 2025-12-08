import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, train_dataset, lr=1e-4, batch_size=8, device="cuda"):
        self.device = device
        self.model = model.to(device)

        self.loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Clasificación entre N ítems → CrossEntropyLoss
        self.loss_fn = nn.CrossEntropyLoss()

        self.opt = optim.Adam(model.parameters(), lr=lr)

        self.loss_history = []  # para graficar

    def train(self, epochs=10):
        action_dim = self.model.head.out_features

        self.model.train()

        for ep in range(epochs):
            total_loss = 0

            for states, actions, rtg in self.loader:
                states = states.to(self.device)            # (B, T)
                actions = actions.to(self.device)          # (B, T)
                rtg = rtg.to(self.device)                  # (B, T, 1)

                pred = self.model(states, actions, rtg)    # (B, T, action_dim)

                loss = self.loss_fn(
                    pred[:, :-1, :].reshape(-1, action_dim), 
                    actions[:, 1:].reshape(-1)
                )

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.loader)
            self.loss_history.append(avg_loss)

            print(f"Epoch {ep+1}/{epochs} – Loss: {avg_loss:.4f}")
