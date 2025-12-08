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
    # states: (B, T)
    # actions: (B, T)
    # rtg: (B, T)

    B, T = states.shape

    # Embeddings
    states_emb = self.state_embedding(states)        # (B, T, d_model)
    actions_emb = self.action_embedding(actions.long())  # (B, T, d_model)
    rtg_emb = self.rtg_embedding(rtg.unsqueeze(-1))  # (B, T, 1) -> emb

    # Sumar embeddings como en Decision Transformer original
    x = states_emb + actions_emb + rtg_emb

    # Transformer
    x = self.transformer(x)          # (B, T, d_model)

    # Predecir próximas acciones
    logits = self.head(x)            # (B, T, num_actions)

    return logits
