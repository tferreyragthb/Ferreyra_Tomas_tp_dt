import torch
from src.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    mrr_at_k,
)

@torch.no_grad()
def evaluate_model(model, dataloader, device="cpu", k=10):
    """
    Evalúa el modelo Decision Transformer usando el TestDataset.

    TestDataset devuelve:
        - items: últimos context_length ítems vistos
        - group: grupo de usuario

    El modelo predice logits sobre el próximo item.
    """

    model.eval()

    all_predictions = []
    all_ground_truth = []

    for batch in dataloader:
        items = batch["items"].to(device)          # (B, context_length)
        groups = batch["group"].to(device)         # (B,)

        # No tenemos returns ni timesteps reales → usamos placeholders
        B, T = items.shape
        rtg = torch.zeros((B, T, 1), device=device)
        timesteps = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)

        # Para evaluación, las "actions" son los mismos items (como en training)
        actions = items.clone()

        # forward
        logits = model(
            states=items,
            actions=actions,
            returns_to_go=rtg,
            timesteps=timesteps,
            user_groups=groups,
        )

        # top-k recommendations
        topk = torch.topk(logits, k=k, dim=-1).indices.cpu().numpy()

        # ground truth: el próximo item real
        # TestDataset NO tiene "targets", pero el último item observado
        # es el que el modelo intenta predecir.
        ground_truth = items[:, -1].cpu().numpy()

        # Guardar por batch
        for preds, gt in zip(topk, ground_truth):
            all_predictions.append(preds.tolist())
            all_ground_truth.append(int(gt))

    # métricas
    metrics = {
        "hit_rate": hit_rate_at_k(all_predictions, all_ground_truth, k),
        "ndcg": ndcg_at_k(all_predictions, all_ground_truth, k),
        "mrr": mrr_at_k(all_predictions, all_ground_truth, k),
    }

    return metrics
