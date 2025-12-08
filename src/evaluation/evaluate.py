import torch
from src.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    mrr_at_k,
)

@torch.no_grad()
def evaluate_model(model, dataloader, device="cpu", k=10):
    """
    Evalúa el modelo sobre un DataLoader de test.

    Retorna un diccionario con:
        - hit_rate
        - ndcg
        - mrr
    """

    model.eval()
    all_predictions = []
    all_ground_truth = []

    for batch in dataloader:
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        rtg = batch["rtg"].to(device)
        timesteps = batch["timesteps"].to(device)
        groups = batch["groups"].to(device)
        targets = batch["targets"].to(device)

        logits = model(
            states=states,
            actions=actions,
            returns_to_go=rtg,
            timesteps=timesteps,
            user_groups=groups,
        )

        topk = torch.topk(logits, k=k, dim=-1).indices.cpu().numpy()
        ground_truth = targets.cpu().numpy()

        for pred_seq, gt_seq in zip(topk, ground_truth):
            all_predictions.append(pred_seq.tolist())
            all_ground_truth.append(gt_seq[0])

    metrics = {
        "hit_rate": hit_rate_at_k(all_predictions, all_ground_truth, k),
        "ndcg": ndcg_at_k(all_predictions, all_ground_truth, k),
        "mrr": mrr_at_k(all_predictions, all_ground_truth, k),
    }

    return metrics
