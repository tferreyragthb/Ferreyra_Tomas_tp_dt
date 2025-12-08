import torch
from torch.nn.functional import softmax
import numpy as np
from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k, mrr_at_k


def evaluate_model(model, dataloader, device, k=10):
    model.eval()

    all_hits = []
    all_ndcg = []
    all_mrr = []

    with torch.no_grad():
        for batch in dataloader:

            items = batch["items"].to(device)         # (B, context_length)
            groups = batch["group"].to(device)        # (B)

            B, T = items.shape

            # Timesteps secuenciales
            timesteps = torch.arange(T, dtype=torch.long, device=device)
            timesteps = timesteps.unsqueeze(0).repeat(B, 1)

            # RTG dummy (valor alto)
            rtg = torch.ones((B, T, 1), dtype=torch.float32, device=device) * 50.0

            # Pasar por el modelo
            logits = model(
                states=items,
                actions=None,
                returns_to_go=rtg,
                timesteps=timesteps,
                groups=groups,
            )

            # logits finales (último paso)
            last_logits = logits[:, -1, :]    # (B, num_items)
            probs = softmax(last_logits, dim=-1)

            # Ground truths reales (último item verdadero)
            true_items = items[:, -1].cpu().numpy()

            # Top-k predicciones
            topk = torch.topk(probs, k, dim=-1).indices.cpu().numpy()

            for i in range(B):
                pred = topk[i]
                true = true_items[i]

                all_hits.append(hit_rate_at_k(true, pred))
                all_ndcg.append(ndcg_at_k(true, pred))
                all_mrr.append(mrr_at_k(true, pred))

    return {
        f"hit_rate@{k}": float(np.mean(all_hits)),
        f"ndcg@{k}": float(np.mean(all_ndcg)),
        f"mrr@{k}": float(np.mean(all_mrr)),
    }
