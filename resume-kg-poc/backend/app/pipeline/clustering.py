"""Step 7 — DBSCAN + K-Means Clustering (paper Sec. VII).

DBSCAN: discovers natural semantic groups + noise/outliers (no k assumed).
K-Means: partitions into k semantic groups (k estimated, default 3).
Both run on the same Sentence-BERT embedding space.
"""
import numpy as np


def cluster(vectors: list[list[float]], entity_texts: list[str]) -> dict:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score

    V = np.array(vectors, dtype=float)
    n = len(V)

    # --- DBSCAN (cosine metric; eps tuned for MiniLM space) ---
    try:
        db = DBSCAN(eps=0.5, min_samples=2, metric="cosine").fit(V)
        db_labels = [int(x) for x in db.labels_]
    except Exception:
        db_labels = [-1] * n
    db_groups: dict[int, list[str]] = {}
    for txt, lab in zip(entity_texts, db_labels):
        db_groups.setdefault(lab, []).append(txt)
    noise = db_groups.pop(-1, [])

    # --- K-Means (k = min(4, n), at least 2 when possible) ---
    k = max(2, min(4, n)) if n >= 2 else 1
    km_labels = [0] * n
    sil = None
    if n >= 2:
        try:
            km = KMeans(n_clusters=min(k, n), n_init=10, random_state=42).fit(V)
            km_labels = [int(x) for x in km.labels_]
            if len(set(km_labels)) > 1 and n > len(set(km_labels)):
                sil = round(float(silhouette_score(V, km_labels)), 4)
        except Exception:
            pass
    km_groups: dict[int, list[str]] = {}
    for txt, lab in zip(entity_texts, km_labels):
        km_groups.setdefault(lab, []).append(txt)

    return {
        "dbscan": {
            "algorithm": "DBSCAN(eps=0.5, min_samples=2, metric=cosine)",
            "purpose": "discover natural semantic groups + noise/outliers",
            "labels": db_labels,
            "clusters": {str(kk): v for kk, v in db_groups.items()},
            "noise_outliers": noise,
            "n_clusters": len(db_groups),
        },
        "kmeans": {
            "algorithm": f"KMeans(k={k})",
            "purpose": "group entities into k semantic clusters",
            "k": k,
            "labels": km_labels,
            "clusters": {str(kk): v for kk, v in km_groups.items()},
            "silhouette_score": sil,
        },
    }
