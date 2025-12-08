import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Estado es un ÍNDICE -> usar EMBEDDING
        self.state_embed = nn.Embedding(state_dim, hidden_dim)

        # Acción también es un ÍNDICE -> usar EMBEDDING
        self.action_embed = nn.Embedding(action_dim, hidden_dim)

        # RTG sí es escalar
        self.rtg_embed = nn.Linear(1, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)

        # predicción de próxima acción (clasificación entre 752 ítems)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, states, actions, rtg):
        """
        states: (B, T)
        actions: (B, T)
        rtg: (B, T, 1)
        """

        s = self.state_embed(states)       # (B, T, H)
        a = self.action_embed(actions)     # (B, T, H)
        r = self.rtg_embed(rtg)            # (B, T, H)

        x = s + a + r

        x = self.transformer(x)
        pred = self.head(x)                # (B, T, action_dim)

        return pred
