import numpy as np
import torch

class PopularityRecommender:
  
    def __init__(self, train_interactions, num_items):
        """
        train_interactions: lista de listas con items usados por cada usuario
        num_items: total de items únicos
        """
        self.num_items = num_items

        # Contar frecuencia por ítem
        counts = np.zeros(num_items)

        for seq in train_interactions:
            for item in seq:
                counts[item] += 1

        # Ordenar de mayor a menor popularidad
        self.popular_items = np.argsort(-counts)

    def recommend(self, k=10):
        """
        Retorna los top-k ítems más populares.
        """
        return self.popular_items[:k]

