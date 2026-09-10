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

    # Any extra CLI args after the query become the simulated session's ACL
    # groups (e.g. "role:principal" or "partner:acme") -- lets you test ACL
    # filtering and permission-sensitive queries without going through the web UI.
    query = sys.argv[1]
    user_groups = sys.argv[2:] or ["public"]
    if "public" not in user_groups:
        user_groups = ["public"] + user_groups

    # Calls the exact same retrieve() that chat.py uses -- this tool skips
    # only the permission-refusal check and the LLM call, not any of the
    # actual retrieval/rerank/gate logic.
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
        flag = " [GUARANTEED -- always included]" if r.is_guaranteed else ""
        print(f"{i:<4} {rerank_str:<7} {r.fused_score:<7.4f} {','.join(r.arms):20} {r.doc_id:28} {r.heading}{flag}")


if __name__ == "__main__":
    main()
