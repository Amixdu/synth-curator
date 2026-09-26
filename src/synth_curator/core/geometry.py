"""Cosine k-NN in feature space (a plain-torch replacement for a FAISS dependency).

Distances match FAISS IndexFlatL2 on L2-normalized vectors: squared L2 = 2 - 2*cos.
Used by the KNN map, realism (synthetic vs real) and diversity (synthetic vs synthetic).
"""
from __future__ import annotations

import torch


def knn_cosine(query: torch.Tensor, db: torch.Tensor, k: int, chunk: int = 2048):
    """Returns (D, I): squared-L2 distances and indices of the k nearest `db` rows per `query` row, ascending."""
    q = query.float()
    d = db.float()
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    d = d / (d.norm(dim=1, keepdim=True) + 1e-8)
    if torch.cuda.is_available():
        d = d.cuda()
    dists, idxs = [], []
    for i in range(0, len(q), chunk):
        sim = q[i:i + chunk].to(d.device) @ d.T
        top = sim.topk(k, dim=1)
        dists.append((2 - 2 * top.values).clamp_min(0).cpu())
        idxs.append(top.indices.cpu())
    return torch.cat(dists).numpy(), torch.cat(idxs).numpy()
