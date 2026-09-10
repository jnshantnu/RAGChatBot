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
from retrieval.confidence import MIN_RERANK_SCORE
from retrieval.pipeline import retrieve

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")

# Maximally permissive groups so recall isn't confounded by ACL filtering --
# ACL correctness is a separate, deliberate test (see README).
ALL_GROUPS = ["public", "partner:acme", "partner:globex", "role:principal", "role:distributor"]


def run_eval():
    # For each golden-set query: run the exact same retrieve() the app uses,
    # then check whether the expected document shows up in the top 5/10
    # results (for answerable queries) or whether the gate correctly abstains
    # (for the deliberately-unanswerable ones). No LLM call here -- this only
    # tests retrieval quality, not generation quality.
    with open(GOLDEN_SET_PATH) as fh:
        golden_set = json.load(fh)

    hits_at_5 = 0
    hits_at_10 = 0
    answerable_count = 0
    abstain_correct = 0
    unanswerable_count = 0
    internal_only_correct = 0
    internal_only_count = 0
    requires_internal_correct = 0
    requires_internal_count = 0

    rows = []
    with psycopg.connect(config.DATABASE_URL) as conn:
        for case in golden_set:
            result = retrieve(conn, case["query"], ALL_GROUPS)

            # Independent of the case's main type below: some queries need
            # internal guidance to score as genuinely relevant (not just
            # present -- it's always present now, see retrieval/pipeline.py),
            # regardless of whether they also have citable grounding.
            if case.get("requires_internal"):
                requires_internal_count += 1
                requires_internal_correct += result.best_internal_score >= MIN_RERANK_SCORE
            # Recall should measure citable grounding, same as what the user
            # actually gets cited -- internal docs (e.g. ADSK-BIZ-RULES.md) can
            # legitimately outrank public chunks for a shared topic without that
            # being a retrieval regression, since pipeline.py gives citable and
            # internal chunks their own top-K slots downstream of this list.
            citable_candidates = [r for r in result.candidates if not r.is_internal]
            doc_ids_top5 = [r.doc_id for r in citable_candidates[:5]]
            doc_ids_top10 = [r.doc_id for r in citable_candidates[:10]]

            if case.get("internal_only"):
                # The correct grounding for this query lives only in an internal
                # doc, which is deliberately excluded from doc_ids_top5/10 above
                # -- there's no citable doc_id to check recall against. The only
                # thing worth verifying is that the gate doesn't abstain.
                internal_only_count += 1
                correct = not result.gate.abstain
                internal_only_correct += correct
                rows.append((case["query"][:50], "internal_only", "-", "-", result.gate.abstain))
            elif case["answerable"]:
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
    print(f"internal-only grounding correctly not abstained: {internal_only_correct}/{internal_only_count}")
    print(f"requires-internal queries scoring internal content relevant: {requires_internal_correct}/{requires_internal_count}")
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
