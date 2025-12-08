import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.rtg_embed = nn.Linear(1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, states, actions, rtg):
        """
        states: (B, T, state_dim)
        actions: (B, T, action_dim)
        rtg: (B, T, 1)
        """
        B, T, _ = states.shape

        s = self.state_embed(states)     # (B, T, H)
        a = self.action_embed(actions)   # (B, T, H)
        r = self.rtg_embed(rtg)          # (B, T, H)

        x = s + a + r                    # (B, T, H)

        x = self.transformer(x)          # (B, T, H)
        pred = self.head(x)              # (B, T, action_dim)

        return pred
