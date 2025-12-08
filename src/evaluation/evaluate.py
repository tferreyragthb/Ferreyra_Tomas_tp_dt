 import torch
from src.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    mrr_at_k,
)

@torch.no_grad()
def evaluate_model(model, dataloader, device="cpu", k=10):
    """
    Evalúa el modelo usando SOLO los últimos context_length items
    (como define TestDataset).

    Cada batch contiene:
        - items: secuencia de entrada (contexto)
        - group: grupo del usuario
        - target: último ítem real
    """

    model.eval()

    all_predictions = []
    all_ground_truth = []

    for batch in dataloader:
        items = batch["items"].to(device)
        groups = batch["group"].to(device)

        # Inferimos sin returns ni timesteps (no se usan en test)
        logits = model(
            states=items,
            actions=items,
            returns_to_go=None,
            timesteps=None,
            user_groups=groups,
        )

        # TOP-K predicciones
        topk = torch.topk(logits, k=k, dim=-1).indices.cpu().numpy()

        # Ground-truth = último item de la secuencia original
        ground_truth = items[:, -1].cpu().numpy()

        for pred_seq, gt in zip(topk, ground_truth):
            all_predictions.append(pred_seq.tolist())
            all_ground_truth.append(int(gt))

    # Métricas
    metrics = {
        "hit_rate": hit_rate_at_k(all_predictions, all_ground_truth, k),
        "ndcg": ndcg_at_k(all_predictions, all_ground_truth, k),
        "mrr": mrr_at_k(all_predictions, all_ground_truth, k),
    }

    return metrics
