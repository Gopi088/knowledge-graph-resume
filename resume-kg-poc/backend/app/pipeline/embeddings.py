"""Step 5 — Semantic Representation with BERT / Sentence-BERT (paper Sec. VII).

Converts each entity (text + type context) and each sentence into dense
vectors. Uses sentence-transformers (all-MiniLM-L6-v2, a Sentence-BERT
model). Falls back to TF-IDF vectors when the model can't be downloaded
(offline PoC) — flagged via `model` field so the distinction is visible.
"""
import numpy as np

_model = None
_model_name = "all-MiniLM-L6-v2"


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_model_name)
        return _model
    except Exception as e:
        print(f"[embeddings] Sentence-BERT unavailable ({e}); using TF-IDF fallback.")
        return None


def embed_texts(texts: list[str]) -> tuple[list[list[float]], str]:
    """Embed arbitrary texts with the same backend as entities.

    Returns (L2-normalized vectors, backend name). Shared by entity
    embeddings and context-aware block detection so both stages live in
    the same semantic space.
    """
    import numpy as np

    model = _load_model()
    if model is not None:
        vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        backend = f"sentence-transformers/{_model_name}"
    else:  # TF-IDF fallback (still L2-normalized, cosine-compatible)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        mat = vectorizer.fit_transform(texts).toarray().astype(float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        vecs = mat / norms
        backend = "tfidf-fallback (offline)"
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype(float).tolist(), backend


def embed_entities(entities: list[dict], sentences: list[str]) -> dict:
    texts = [f"{e['text']} ({e['type']})" for e in entities]
    vecs, backend = embed_texts(texts) if texts else ([], "tfidf-fallback (offline)")
    sent_vecs, _ = embed_texts(sentences) if sentences else ([], backend)
    return {
        "model": backend,
        "dim": int(len(vecs[0])) if vecs else 0,
        "entity_texts": texts,
        "vectors": vecs,  # full precision for research use (already L2-normalized)
        "vectors_preview": [v[:8] for v in vecs],
        "sentence_vectors_preview": [v[:8] for v in sent_vecs[:3]],
    }
