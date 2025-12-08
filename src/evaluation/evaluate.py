import torch
from src.evaluation.metrics import hit_rate, ndcg, mrr

@torch.no_grad()
def evaluate_model(model, dataloader, device="cpu", k=10):
    """
    Evalúa el modelo en un dataloader (solo test).
    Retorna promedios de HR, NDCG y MRR.
    """

    model.eval()
    hr_list, ndcg_list, mrr_list = [], [], []

    for states, actions, rtg in dataloader:

        states = states.to(device)
        actions = actions.to(device)
        rtg = rtg.to(device)

        # Predicción completa
        pred = model(states, actions, rtg)   # (B, T, num_items)

        # Última posición
        last_pred = pred[:, -1, :]           # (B, num_items)

        # Top-k predicciones
        topk = torch.topk(last_pred, k).indices.cpu().tolist()

        # Target real (la última acción verdadera)
        target = actions[:, -1].cpu().tolist()

        # Calcular métricas instancia por instancia
        for pk, tg in zip(topk, target):
            hr_list.append(hit_rate(pk, tg))
            ndcg_list.append(ndcg(pk, tg))
            mrr_list.append(mrr(pk, tg))

    return {
        "hit_rate": sum(hr_list) / len(hr_list),
        "ndcg": sum(ndcg_list) / len(ndcg_list),
        "mrr": sum(mrr_list) / len(mrr_list),
    }

