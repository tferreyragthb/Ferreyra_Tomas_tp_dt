import numpy as np

def hit_rate(preds, target):
    """
    preds: lista ordenada de ítems recomendados (top-k)
    target: ítem real que ocurrió
    """
    return 1.0 if target in preds else 0.0


def ndcg(preds, target):
    """
    NDCG@k donde target es el ítem verdadero.
    """
    if target in preds:
        index = preds.index(target)
        return 1.0 / np.log2(index + 2)
    return 0.0


def mrr(preds, target):
    """
    MRR@k
    """
    if target in preds:
        index = preds.index(target)
        return 1.0 / (index + 1)
    return 0.0

