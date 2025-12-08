import numpy as np

def hit_rate_at_k(predictions, ground_truth, k=10):
    """
    predictions: lista de listas -> items recomendados ordenados
    ground_truth: lista de items verdaderos (1 por usuario)
    """
    hits = 0
    for pred, gt in zip(predictions, ground_truth):
        if gt in pred[:k]:
            hits += 1
    return hits / len(ground_truth)


def ndcg_at_k(predictions, ground_truth, k=10):
    """
    Calcula NDCG@K.
    """
    ndcg_total = 0.0

    for pred, gt in zip(predictions, ground_truth):
        if gt in pred[:k]:
            rank = pred.index(gt)
            ndcg_total += 1 / np.log2(rank + 2)  # +2 porque rank empieza en 0

    return ndcg_total / len(ground_truth)


def mrr_at_k(predictions, ground_truth, k=10):
    """
    Mean Reciprocal Rank @ K.
    """
    mrr_total = 0.0

    for pred, gt in zip(predictions, ground_truth):
        if gt in pred[:k]:
            rank = pred.index(gt)
            mrr_total += 1.0 / (rank + 1)

    return mrr_total / len(ground_truth)
