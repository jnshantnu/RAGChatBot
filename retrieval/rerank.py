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

_model = None  # module-level cache: loaded once per process, reused across every request in that process


def _get_model():
    # Lazy import + lazy load: sentence-transformers/torch are heavy to import,
    # and loading the model takes a few seconds -- both only need to happen
    # once, on first use, not at module import time (which would slow down
    # every script that merely imports this module, even ones that never rerank).
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME)
    return _model


def _sigmoid(x: float) -> float:
    # CrossEncoder.predict() returns raw, unbounded logits; squashing through a
    # sigmoid turns them into a 0-1 relevance probability that's actually
    # interpretable (and comparable to a fixed threshold in confidence.py).
    return 1.0 / (1.0 + math.exp(-x))


def rerank(query: str, fused_results: list[FusedResult], top_n: int = RERANK_CANDIDATES) -> list[FusedResult]:
    """Returns a new list, re-sorted by rerank_score (raw cross-encoder logits
    passed through a sigmoid so scores read as a 0-1 relevance probability,
    matching the case study's own example scores of 0.94/0.81/0.44/0.19)."""
    # Only rerank the top slice of the fused list -- the cross-encoder is much
    # slower per-item than the bi-encoder/keyword arms, so this is the
    # "rerank-narrow" half of "retrieve-wide, rerank-narrow".
    candidates = fused_results[:top_n]
    if not candidates:
        return []

    model = _get_model()
    # (query, chunk_text) pairs -- unlike embedding, the cross-encoder reads
    # both together in one forward pass, which is what makes it more accurate
    # than comparing two separately-computed embedding vectors.
    pairs = [(query, c.chunk_text) for c in candidates]
    raw_scores = model.predict(pairs)

    # Mutates rerank_score on the existing FusedResult objects (rather than
    # building new ones), since these same objects flow onward into
    # RetrievalResult.candidates/top_k and eventually the API response.
    for candidate, raw_score in zip(candidates, raw_scores):
        candidate.rerank_score = _sigmoid(float(raw_score))

    return sorted(candidates, key=lambda r: r.rerank_score, reverse=True)
