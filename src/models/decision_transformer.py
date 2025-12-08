import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Para ítems representados como enteros (0..751)
        self.state_embed = nn.Embedding(state_dim, hidden_dim)
        self.action_embed = nn.Embedding(action_dim, hidden_dim)

        # RTG sí es numérico (B, T, 1)
        self.rtg_embed = nn.Linear(1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Predecir acciones: logits (B, T, action_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, states, actions, rtg):
    """
    states: (B, T) — índices de ítems
    actions: (B, T) — índices de ítems
    rtg: (B, T) — returns to go
    """

    B, T = states.shape

    # RTG necesita un canal extra
    if rtg.dim() == 2:
        rtg = rtg.unsqueeze(-1)   # (B, T, 1)

    rtg = rtg.float()             

    # Embeddings
    s = self.state_embed(states)       # (B, T, H)
    a = self.action_embed(actions)     # (B, T, H)
    r = self.rtg_embed(rtg)            # (B, T, H)

    # Combinación
    x = s + a + r                      # (B, T, H)

    # Transformer
    x = self.transformer(x)            # (B, T, H)

    # Predicción
    pred = self.head(x)                # (B, T, action_dim)

    return pred

