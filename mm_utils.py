import numpy as np
import torch
import umap
import pandas as pd

# ===================== Batching =====================
def batch_infer(model, processor, inputs, batch_size=16, mode="text"):
    outputs = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        if mode == "text":
            enc = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            out = model.get_text_features(**enc).cpu().numpy()
        elif mode == "image":
            enc = processor(images=batch, return_tensors="pt")
            out = model.get_image_features(**enc).cpu().numpy()
        outputs.append(out)
    return np.vstack(outputs)

# ===================== Retrieval Metrics =====================
def precision_at_k(sim_matrix, k=5):
    ranks = np.argsort(-sim_matrix, axis=1)
    precs = [(1 if i in ranks[i, :k] else 0) for i in range(sim_matrix.shape[0])]
    return np.mean(precs)

def recall_at_k(sim_matrix, k=5):
    ranks = np.argsort(-sim_matrix, axis=1)
    recs = [(1 if i in ranks[i, :k] else 0) for i in range(sim_matrix.shape[0])]
    return np.mean(recs)

def mrr(sim_matrix):
    ranks = np.argsort(-sim_matrix, axis=1)
    recip = []
    for i in range(sim_matrix.shape[0]):
        pos = np.where(ranks[i] == i)[0][0] + 1
        recip.append(1.0 / pos)
    return np.mean(recip)

# ===================== Embedding Visualization =====================
def umap_projection(t_emb, i_emb, n_neighbors=10, min_dist=0.1):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=42)
    emb_2d = reducer.fit_transform(np.vstack([t_emb, i_emb]))
    labels = ["text"] * len(t_emb) + ["image"] * len(i_emb)
    return pd.DataFrame({"x": emb_2d[:, 0], "y": emb_2d[:, 1], "type": labels})
