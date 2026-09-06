"""Reciprocal Rank Fusion: fuses on rank, not raw score, so cosine distance
and ts_rank_cd (different, non-comparable scales) never need normalising."""
from dataclasses import dataclass

from retrieval.hybrid_search import Candidate

K = 60


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
        return bool((self.metadata or {}).get("internal"))


def reciprocal_rank_fusion(keyword_results: list[Candidate], semantic_results: list[Candidate]) -> list[FusedResult]:
    fused: dict[str, FusedResult] = {}

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
