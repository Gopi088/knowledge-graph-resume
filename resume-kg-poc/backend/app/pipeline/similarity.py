"""Step 6 — Cosine Similarity (paper: closeness between document/entity vectors).

Pairwise cosine similarity over L2-normalized entity embeddings.
Used ONLY to measure semantic closeness — high similarity does NOT
create a factual edge (per paper distinction + user requirement).
Similarity edges are labelled SEMANTICALLY_SIMILAR / semantic-inferred.
"""
import numpy as np


def cosine_similarity_matrix(vectors: list[list[float]]) -> dict:
    V = np.array(vectors, dtype=float)
    sim = V @ V.T  # already L2-normalized
    sim = np.clip(sim, -1.0, 1.0)
    pairs = []
    n = sim.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({"i": i, "j": j, "similarity": round(float(sim[i, j]), 4)})
    pairs.sort(key=lambda p: -p["similarity"])
    return {
        "measure": "cosine similarity over Sentence-BERT embeddings",
        "matrix": sim.tolist(),
        "top_pairs": pairs[:25],
        "pair_count": len(pairs),
    }
