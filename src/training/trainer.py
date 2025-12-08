import torch
import torch.nn as nn

def train_decision_transformer(
    model,
    train_loader,
    optimizer,
    device="cpu",
    num_epochs=10,
):
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    history = []

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in train_loader:

            # === NUEVAS KEYS DEL DATASET ===
            items = batch["items"].to(device)           # (B, L)
            returns_to_go = batch["returns_to_go"].to(device)  # (B, L)
            timesteps = batch["timesteps"].to(device)   # (B, L)
            groups = batch["group"].to(device)          # (B,)

            # Targets: predecir el siguiente item
            targets = items[:, 1:].clone()              # shift
            targets = torch.cat(
                [targets, torch.full((targets.size(0), 1), -1, device=device, dtype=torch.long)],
                dim=1
            )

            # Forward
            logits = model(
                states=items,
                actions=items,
                returns_to_go=returns_to_go.unsqueeze(-1),
                timesteps=timesteps,
                user_groups=groups,
            )

            # logits: (B, L, num_items)
            # reshape para CrossEntropyLoss
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)

            loss = loss_fn(logits_flat, targets_flat)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    return model, history
