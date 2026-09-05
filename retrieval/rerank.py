"""Cross-encoder rerank: judges query and chunk together for a real relevance
score, instead of the bi-encoder's geometric proxy (comparing two independently
computed vectors by angle). Runs on the top ~20 fused candidates, not the whole
corpus -- retrieve-wide with the cheap bi-encoder/keyword arms, rerank-narrow
with the expensive cross-encoder. If the model can't be loaded, callers fall
back to RRF order and flag `degraded.rerank` rather than pretending it ran.
"""
import math

from retrieval.rrf import FusedResult

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME)
    return _model


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank(query: str, fused_results: list[FusedResult], top_n: int = RERANK_CANDIDATES) -> list[FusedResult]:
    """Returns a new list, re-sorted by rerank_score (raw cross-encoder logits
    passed through a sigmoid so scores read as a 0-1 relevance probability,
    matching the case study's own example scores of 0.94/0.81/0.44/0.19)."""
    candidates = fused_results[:top_n]
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c.chunk_text) for c in candidates]
    raw_scores = model.predict(pairs)

    for candidate, raw_score in zip(candidates, raw_scores):
        candidate.rerank_score = _sigmoid(float(raw_score))

    return sorted(candidates, key=lambda r: r.rerank_score, reverse=True)
