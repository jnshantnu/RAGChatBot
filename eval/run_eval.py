"""Tiny stand-in for the case study's eval harness: recall@5/@10 against a
golden set, plus abstain correctness on deliberately unanswerable queries.
Run after any change to chunking, embeddings, or retrieval to catch
regressions before they reach the UI.
"""
import json
import os
import sys

import psycopg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from retrieval.pipeline import retrieve

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")

# Maximally permissive groups so recall isn't confounded by ACL filtering --
# ACL correctness is a separate, deliberate test (see README).
ALL_GROUPS = ["public", "partner:acme", "partner:globex", "role:principal", "role:distributor"]


def run_eval():
    with open(GOLDEN_SET_PATH) as fh:
        golden_set = json.load(fh)

    hits_at_5 = 0
    hits_at_10 = 0
    answerable_count = 0
    abstain_correct = 0
    unanswerable_count = 0

    rows = []
    with psycopg.connect(config.DATABASE_URL) as conn:
        for case in golden_set:
            result = retrieve(conn, case["query"], ALL_GROUPS)
            # Recall should measure citable grounding, same as what the user
            # actually gets cited -- internal docs (e.g. ADSK-BIZ-RULES.md) can
            # legitimately outrank public chunks for a shared topic without that
            # being a retrieval regression, since pipeline.py gives citable and
            # internal chunks their own top-K slots downstream of this list.
            citable_candidates = [r for r in result.candidates if not r.is_internal]
            doc_ids_top5 = [r.doc_id for r in citable_candidates[:5]]
            doc_ids_top10 = [r.doc_id for r in citable_candidates[:10]]

            if case["answerable"]:
                answerable_count += 1
                expected = case["expected_doc_ids"]
                hit5 = any(doc_id in doc_ids_top5 for doc_id in expected)
                hit10 = any(doc_id in doc_ids_top10 for doc_id in expected)
                hits_at_5 += hit5
                hits_at_10 += hit10
                rows.append((case["query"][:50], "answerable", hit5, hit10, result.gate.abstain))
            else:
                unanswerable_count += 1
                correct = result.gate.abstain
                abstain_correct += correct
                rows.append((case["query"][:50], "unanswerable", "-", "-", result.gate.abstain))

    print(f"{'query':52} {'type':13} {'hit@5':6} {'hit@10':7} abstained")
    for q, kind, h5, h10, abstained in rows:
        print(f"{q:52} {kind:13} {str(h5):6} {str(h10):7} {abstained}")

    print()
    print(f"recall@5:  {hits_at_5}/{answerable_count}")
    print(f"recall@10: {hits_at_10}/{answerable_count}")
    print(f"retrieval-gate abstain rate on unanswerable: {abstain_correct}/{unanswerable_count}")
    print(
        "note: this only measures the fast, pre-LLM gate (retrieval/confidence.py). "
        "Queries it doesn't catch still get a real LLM call, whose system prompt "
        "instructs it to say \"I don't have that information\" when the retrieved "
        "context doesn't support an answer -- the two-layer defense described in "
        "the case study (gate skips the LLM on clear misses; grounding is the "
        "backstop on subtler content gaps). See chat.py's full pipeline for the "
        "end-to-end behavior this harness doesn't exercise."
    )


if __name__ == "__main__":
    run_eval()
