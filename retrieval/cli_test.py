"""Standalone retrieval inspector -- no LLM call, just the SQL + fusion + gate.
Usage: python -m retrieval.cli_test "your question" partner:acme role:principal
"""
import os
import sys

import psycopg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from retrieval.pipeline import retrieve


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m retrieval.cli_test "your question" [group1 group2 ...]')
        sys.exit(1)

    query = sys.argv[1]
    user_groups = sys.argv[2:] or ["public"]
    if "public" not in user_groups:
        user_groups = ["public"] + user_groups

    with psycopg.connect(config.DATABASE_URL) as conn:
        result = retrieve(conn, query, user_groups)

    print(f"query: {query!r}")
    print(f"user_groups: {user_groups}")
    print(f"timings_ms: {result.timings_ms}")
    print(f"degraded_rerank: {result.degraded_rerank}")
    print(f"gate: abstain={result.gate.abstain} reason={result.gate.reason} best_score={result.gate.best_score:.4f}")
    print()
    print(f"{'rank':4} {'rerank':7} {'fused':7} {'arms':20} {'doc_id':28} heading")
    for i, r in enumerate(result.candidates[:10], start=1):
        rerank_str = f"{r.rerank_score:.4f}" if r.rerank_score is not None else "-"
        flag = " [INTERNAL -- never citable/exposed]" if r.is_internal else ""
        print(f"{i:<4} {rerank_str:<7} {r.fused_score:<7.4f} {','.join(r.arms):20} {r.doc_id:28} {r.heading}{flag}")


if __name__ == "__main__":
    main()
