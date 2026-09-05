"""Confidence gate: abstain if retrieval signal is weak, decided AFTER retrieval
and independent of permissions. 'I don't have that' is a valid output and the
primary hallucination control.

Reads the cross-encoder's rerank_score when available (a real, calibrated
0-1 relevance probability -- the case study's own gate reads exactly this).
Falls back to raw semantic cosine similarity, with keyword-arm corroboration
for borderline cases, only in `degraded.rerank` mode. The fallback path exists
because we measured it fail on some queries when the embedding model changed
(see README) -- it's a documented weaker substitute, not an equivalent signal.
"""
from dataclasses import dataclass

from retrieval.hybrid_search import Candidate
from retrieval.rrf import FusedResult

# Rerank-score thresholds: the cross-encoder's sigmoid output is a genuine
# relevance probability, so these read like the case study's own example scores.
MIN_RERANK_SCORE = 0.30

# Fallback (degraded.rerank) thresholds: raw cosine similarity, tuned per
# embedding model against eval/golden_set.json -- re-tune whenever the
# embedding model or corpus changes. Documented weak spot, see README.
MIN_SEMANTIC_SIMILARITY = 0.50
BORDERLINE_SIMILARITY = 0.40


@dataclass
class GateResult:
    abstain: bool
    reason: str
    best_score: float


def confidence_gate(
    fused_results: list[FusedResult],
    semantic_candidates: list[Candidate],
    keyword_candidates: list[Candidate],
    reranked: bool,
) -> GateResult:
    if not fused_results:
        return GateResult(abstain=True, reason="no_candidates", best_score=0.0)

    if reranked:
        best_rerank = fused_results[0].rerank_score or 0.0
        if best_rerank < MIN_RERANK_SCORE:
            return GateResult(abstain=True, reason="weak_rerank_score", best_score=best_rerank)
        return GateResult(abstain=False, reason="ok", best_score=best_rerank)

    best_similarity = 1 - min((c.score for c in semantic_candidates), default=1.0)
    keyword_matched = len(keyword_candidates) > 0

    if best_similarity >= MIN_SEMANTIC_SIMILARITY:
        return GateResult(abstain=False, reason="ok_degraded", best_score=best_similarity)

    if best_similarity >= BORDERLINE_SIMILARITY and keyword_matched:
        return GateResult(abstain=False, reason="ok_keyword_corroborated_degraded", best_score=best_similarity)

    return GateResult(abstain=True, reason="weak_semantic_similarity_degraded", best_score=best_similarity)
