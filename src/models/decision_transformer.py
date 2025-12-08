import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        max_len=50,
        n_layers=3,
        n_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.max_len = max_len

        # Embeddings
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.rtg_embed = nn.Linear(1, hidden_dim)

        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output: next action
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, states, actions, rtg):
        """
        states:  (B, T, state_dim)
        actions: (B, T, action_dim)
        rtg:     (B, T, 1)
        """
        B, T, _ = states.shape

        x = (
            self.state_embed(states)
            + self.action_embed(actions)
            + self.rtg_embed(rtg)
            + self.pos_encoding[:, :T, :]
        )

        # Transformer
        h = self.transformer(x)

        # Predicción del próximo action
        return self.head(h)

