import torch
import torch.nn as nn


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        num_items: int = 752,
        num_groups: int = 8,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        context_length: int = 20,
        max_timestep: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.num_items = num_items
        self.num_groups = num_groups

        # === EMBEDDINGS ===

        # Items (para history y acciones)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)

        # Grupo de usuario
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)

        # Return-to-go (escalar continuo)
        self.rtg_embedding = nn.Linear(1, hidden_dim)

        # Timestep (positional)
        self.timestep_embedding = nn.Embedding(max_timestep, hidden_dim)

        # === TRANSFORMER ===

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # === PREDICTION HEAD ===

        self.predict_item = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items),
        )

        # LayerNorm final
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        states,        # (B, L)
        actions,       # (B, L)
        returns_to_go, # (B, L, 1)
        timesteps,     # (B, L)
        user_groups,   # (B,)
        attention_mask=None,
    ):
        """
        Devuelve logits sobre items: (B, L, num_items)
        """
        batch_size, seq_len = states.shape

        # === EMBEDDINGS ===

        state_emb = self.item_embedding(states)      # (B, L, H)
        action_emb = self.item_embedding(actions)    # (B, L, H)
        rtg_emb = self.rtg_embedding(returns_to_go)  # (B, L, H)
        time_emb = self.timestep_embedding(timesteps)  # (B, L, H)

        group_emb = self.group_embedding(user_groups).unsqueeze(1)  # (B, 1, H)
        group_emb = group_emb.expand(-1, seq_len, -1)               # (B, L, H)

        # Sumamos todas las fuentes de información
        h = state_emb + rtg_emb + time_emb + group_emb
        h = self.ln(h)

        # === CAUSAL MASK ===
        if attention_mask is None:
            attention_mask = self._generate_causal_mask(seq_len).to(h.device)

        # === TRANSFORMER ===
        h = self.transformer(h, mask=attention_mask)  # (B, L, H)

        # === PREDICCIÓN DE ÍTEM ===
        item_logits = self.predict_item(h)  # (B, L, num_items)

        return item_logits

    def _generate_causal_mask(self, seq_len: int):
        """
        Máscara causal para que cada posición vea solo el pasado.
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len) * float("-inf"),
            diagonal=1,
        )
        return mask
