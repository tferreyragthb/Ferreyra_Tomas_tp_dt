import numpy as np

def normalize_item_ids(df_train):
    """
    Convierte los item_ids arbitrarios (ej: 472, 338, 510, ...)
    a IDs consecutivos desde 0 ... N-1
    """

    # obtener todos los items únicos
    all_items = set()
    for seq in df_train["items"]:
        all_items.update(seq)

    all_items = sorted(all_items)
    mapping = {old: new for new, old in enumerate(all_items)}

    # aplicar el mapeo
    df_train["items"] = df_train["items"].apply(
        lambda seq: np.array([mapping[i] for i in seq])
    )

    return mapping, len(all_items)
