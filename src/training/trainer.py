import torch
import torch.nn.functional as F


def train_decision_transformer(
    model,
    train_loader,
    optimizer,
    device: str = "cuda",
    num_epochs: int = 10,
):
    """
    Entrena el Decision Transformer.

    Args:
        model: instancia de DecisionTransformer
        train_loader: DataLoader que devuelve batches con keys:
            'states', 'actions', 'rtg', 'timesteps', 'groups', 'targets'
        optimizer: optimizer de PyTorch (ej: Adam)
        device: 'cuda' o 'cpu'
        num_epochs: número de épocas

    Returns:
        model: modelo entrenado
        history: lista con loss promedio por época
    """
    model.to(device)
    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            states = batch["states"].to(device)      # (B, L)
            actions = batch["actions"].to(device)    # (B, L)
            rtg = batch["rtg"].to(device)            # (B, L, 1)
            timesteps = batch["timesteps"].to(device)# (B, L)
            groups = batch["groups"].to(device)      # (B,)
            targets = batch["targets"].to(device)    # (B, L)

            # Forward
            logits = model(states, actions, rtg, timesteps, groups)  # (B, L, num_items)

            # Cross-entropy sobre todos los timesteps
            loss = F.cross_entropy(
                logits.reshape(-1, model.num_items),
                targets.reshape(-1),
                ignore_index=-1,  # ignorar padding
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model, history
