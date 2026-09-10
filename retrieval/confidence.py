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

# Rerank-score threshold. Originally 0.30, lowered to 0.20 after finding real,
# correct matches scoring as low as 0.23 for genuinely-phrased-but-generic
# questions ("which APIs are available to me?") -- the cross-encoder reads as
# less confident on vague phrasing even when it found the right chunk.
# Verified against both known-correct abstains in the golden set (0.0017,
# 0.0043) -- nowhere near this bar, so lowering it doesn't let noise through.
MIN_RERANK_SCORE = 0.20

# Dominance rescue: some real matches score low even by the lowered bar above
# -- as low as 0.03 -- when the question is phrased very generically ("which
# business model is applicable to me?"). A genuinely low score is also what
# pure noise looks like, so absolute score alone can't tell them apart down
# there. What can: how far ahead the top result is from the runner-up. An
# unanswerable query's candidates are all noise, clustered close together
# (e.g. 0.0017 vs 0.0004, ~4x); a real match buried by vague phrasing still
# stands out sharply from the noise around it (e.g. 0.0311 vs 0.0001, ~300x).
# Below MIN_RERANK_SCORE, rescue only if the top score clears a low floor AND
# dominates the next-best by a wide margin -- both conditions, so neither a
# low floor nor a large ratio alone can rescue a fully noisy result set.
DOMINANT_RESCUE_FLOOR = 0.01
DOMINANT_RESCUE_RATIO = 10

# Fallback (degraded.rerank) thresholds: raw cosine similarity, tuned per
# embedding model against eval/golden_set.json -- re-tune whenever the
# embedding model or corpus changes. Documented weak spot, see README.
MIN_SEMANTIC_SIMILARITY = 0.50
BORDERLINE_SIMILARITY = 0.40


@dataclass
class GateResult:
    abstain: bool
    reason: str   # short machine-readable reason code, e.g. "weak_rerank_score" -- useful in logs/debug output
    best_score: float


def confidence_gate(
    fused_results: list[FusedResult],
    semantic_candidates: list[Candidate],
    keyword_candidates: list[Candidate],
    reranked: bool,   # False when retrieval/pipeline.py's rerank() call failed -- switches to the fallback path below
) -> GateResult:
    # Nothing came back from either search arm at all.
    if not fused_results:
        return GateResult(abstain=True, reason="no_candidates", best_score=0.0)

    # Normal path: trust the cross-encoder's calibrated relevance score for the
    # single best-ranked result. This is the only path that runs when
    # reranking succeeds, which is the overwhelming majority of requests.
    if reranked:
        best_rerank = fused_results[0].rerank_score or 0.0
        if best_rerank >= MIN_RERANK_SCORE:
            return GateResult(abstain=False, reason="ok", best_score=best_rerank)

        # Below the main bar -- check the dominance rescue before giving up.
        second_best = (fused_results[1].rerank_score or 0.0) if len(fused_results) > 1 else 0.0
        is_dominant = best_rerank >= DOMINANT_RESCUE_FLOOR and (
            second_best <= 0 or best_rerank / second_best >= DOMINANT_RESCUE_RATIO
        )
        if is_dominant:
            return GateResult(abstain=False, reason="ok_dominant_top1", best_score=best_rerank)

        return GateResult(abstain=True, reason="weak_rerank_score", best_score=best_rerank)

    # Fallback path (reranker unavailable): no calibrated score to read, so
    # fall back to raw cosine similarity from the semantic arm, with a keyword
    # hit allowed to corroborate a borderline (but not hopeless) similarity.
    best_similarity = 1 - min((c.score for c in semantic_candidates), default=1.0)
    keyword_matched = len(keyword_candidates) > 0

    if best_similarity >= MIN_SEMANTIC_SIMILARITY:
        return GateResult(abstain=False, reason="ok_degraded", best_score=best_similarity)

    if best_similarity >= BORDERLINE_SIMILARITY and keyword_matched:
        return GateResult(abstain=False, reason="ok_keyword_corroborated_degraded", best_score=best_similarity)

    return GateResult(abstain=True, reason="weak_semantic_similarity_degraded", best_score=best_similarity)
