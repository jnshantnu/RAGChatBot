"""Cross-encoder rerank: judges query and chunk together for a real relevance
score, instead of the bi-encoder's geometric proxy (comparing two independently
computed vectors by angle). Runs on the top ~20 fused candidates, not the whole
corpus -- retrieve-wide with the cheap bi-encoder/keyword arms, rerank-narrow
with the expensive cross-encoder. If the model can't be loaded, callers fall
back to RRF order and flag `degraded.rerank` rather than pretending it ran.
"""
import math
import threading

from retrieval.rrf import FusedResult

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20

# How much of each chunk the cross-encoder reads, in tokens. None = the model's
# own default (512) -- what "Baseline" mode uses. TUNED_MAX_LENGTH is what
# "Optimized" mode uses. Cost grows with input length, and this corpus's
# chunks are page-sized (~1,500 chars, ~400 tokens), so capping the input is
# the biggest single lever on rerank time: measured ~2x faster at 256.
# Validated against 31 questions (20 golden + 11 role-based, through the
# rewriter), using the full-length reranker as the relevance judge:
#   - golden recall@5/@10, abstain decisions, guaranteed-content checks: all
#     identical to full length;
#   - the top-5 keeps ~97.7% of the relevance (NDCG@5 = 0.977) -- but not all
#     of it, and individual queries can pick a noticeably weaker #1 chunk;
#   - 256 is the largest step that keeps the confidence-gate calibration
#     intact: noise-question scores rise only slightly (phone-number question
#     0.004 -> 0.007, still under confidence.py's 0.01 rescue floor), while at
#     128 they cross it and one correct abstain flips to an answer.
# Cutting RERANK_CANDIDATES instead was rejected: 20 -> 10 passes the golden
# set (14/14) yet loses ~10% of relevance (NDCG@5 0.896).
TUNED_MAX_LENGTH = 256
DEFAULT_MAX_LENGTH = 512  # what CrossEncoder(MODEL_NAME) reads per chunk when no max_length is passed; shown in the debug trace

_models: dict = {}  # keyed by max_length; each variant loaded once per process, reused across requests
_load_lock = threading.Lock()  # app/serve.py preloads in a background thread while a visitor's request may ask for the same model


def _get_model(max_length: int | None = None):
    # Lazy import + lazy load: sentence-transformers/torch are heavy to import,
    # and loading the model takes a few seconds -- both only need to happen
    # once, on first use, not at module import time (which would slow down
    # every script that merely imports this module, even ones that never rerank).
    if max_length not in _models:
        with _load_lock:
            if max_length not in _models:  # re-check: another thread may have loaded it while we waited
                from sentence_transformers import CrossEncoder

                _models[max_length] = (
                    CrossEncoder(MODEL_NAME, max_length=max_length) if max_length else CrossEncoder(MODEL_NAME)
                )
    return _models[max_length]


def _sigmoid(x: float) -> float:
    # CrossEncoder.predict() returns raw, unbounded logits; squashing through a
    # sigmoid turns them into a 0-1 relevance probability that's actually
    # interpretable (and comparable to a fixed threshold in confidence.py).
    return 1.0 / (1.0 + math.exp(-x))


def rerank(
    query: str,
    fused_results: list[FusedResult],
    top_n: int = RERANK_CANDIDATES,
    max_length: int | None = None,
) -> list[FusedResult]:
    """Returns a new list, re-sorted by rerank_score (raw cross-encoder logits
    passed through a sigmoid so scores read as a 0-1 relevance probability,
    matching the case study's own example scores of 0.94/0.81/0.44/0.19).
    `max_length` caps how many tokens of each chunk the model reads -- see
    TUNED_MAX_LENGTH for why and what it costs."""
    # Only rerank the top slice of the fused list -- the cross-encoder is much
    # slower per-item than the bi-encoder/keyword arms, so this is the
    # "rerank-narrow" half of "retrieve-wide, rerank-narrow".
    candidates = fused_results[:top_n]
    if not candidates:
        return []

    model = _get_model(max_length)
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
