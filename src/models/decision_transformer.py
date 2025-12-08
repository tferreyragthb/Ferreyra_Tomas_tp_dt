import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
    ):
        super().__init__()

        self.state_embedding = nn.Embedding(state_dim, hidden_dim)
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        self.rtg_embedding = nn.Linear(1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, states, actions, rtg):
        """
        states: (B, T)
        actions: (B, T)
        rtg: (B, T)
        """

        # ---- Shapes ----
        B, T = states.shape

        # ---- Embeddings ----
        states_emb = self.state_embedding(states)               # (B, T, d)
        actions_emb = self.action_embedding(actions.long())     # (B, T, d)
        rtg_emb = self.rtg_embedding(rtg.unsqueeze(-1))         # (B, T, 1) -> (B, T, d)

        # ---- Combine embeddings ----
        x = states_emb + actions_emb + rtg_emb                  # (B, T, d)

        # ---- Transformer ----
        x = self.transformer(x)                                 # (B, T, d)

        # ---- Output logits ----
        logits = self.head(x)                                   # (B, T, action_dim)

        return logits
