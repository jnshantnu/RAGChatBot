"""Reciprocal Rank Fusion: fuses on rank, not raw score, so cosine distance
and ts_rank_cd (different, non-comparable scales) never need normalising."""
from dataclasses import dataclass

from retrieval.hybrid_search import Candidate

K = 60  # standard RRF constant; dampens the impact of rank 1 vs rank 2 so one arm can't dominate alone


# One chunk's merged result after fusing the keyword and semantic rankings.
# Starts life here with just fused_score/arms; retrieval/rerank.py fills in
# rerank_score later, in place, on these same objects.
@dataclass
class FusedResult:
    chunk_id: str
    doc_id: str
    heading: str
    chunk_text: str
    fused_score: float
    arms: list[str]  # which arm(s) contributed: "keyword", "semantic"
    metadata: dict = None
    rerank_score: float | None = None  # set by retrieval/rerank.py; None if rerank was skipped

    @property
    def is_internal(self) -> bool:
        # Single source of truth for "is this an internal-guidance chunk" --
        # chat.py, pipeline.py, and cli_test.py all check this instead of
        # each re-reading metadata.get("internal") themselves.
        return bool((self.metadata or {}).get("internal"))


def reciprocal_rank_fusion(keyword_results: list[Candidate], semantic_results: list[Candidate]) -> list[FusedResult]:
    # Keyed by chunk_id so a chunk that appears in both arms accumulates score
    # from both instead of creating two separate entries.
    fused: dict[str, FusedResult] = {}

    # First pass: every keyword-arm hit gets its RRF contribution (1/(K+rank)).
    for rank, cand in enumerate(keyword_results, start=1):
        fused[cand.chunk_id] = FusedResult(
            chunk_id=cand.chunk_id,
            doc_id=cand.doc_id,
            heading=cand.heading,
            chunk_text=cand.chunk_text,
            fused_score=1.0 / (K + rank),
            arms=["keyword"],
            metadata=cand.metadata,
        )

    # Second pass: semantic-arm hits either add to an existing entry (chunk
    # found by both arms -- the strongest signal) or create a new one.
    for rank, cand in enumerate(semantic_results, start=1):
        contribution = 1.0 / (K + rank)
        if cand.chunk_id in fused:
            fused[cand.chunk_id].fused_score += contribution
            fused[cand.chunk_id].arms.append("semantic")
        else:
            fused[cand.chunk_id] = FusedResult(
                chunk_id=cand.chunk_id,
                doc_id=cand.doc_id,
                heading=cand.heading,
                chunk_text=cand.chunk_text,
                fused_score=contribution,
                arms=["semantic"],
                metadata=cand.metadata,
            )

    return sorted(fused.values(), key=lambda r: r.fused_score, reverse=True)
