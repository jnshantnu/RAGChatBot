"""Evaluate the query-understanding stage.

    python -m eval.run_understanding_eval
        Classification + normalization accuracy against eval/query_understanding_set.json.
        Pure rules -- no database, no network, runs in about a second.

    python -m eval.run_understanding_eval --retrieval
        ALSO compares retrieval under three query plans, for every golden-set
        question (recall@5, gate decision) and every labeled-set question
        (top rerank score, gate decision):
          legacy   rewrite_query() alone -- how the app behaved before this stage
          planned  chat.plan_query()      -- what production runs now
          phrase   planned + the vocabulary's opt-in API-role phrase appended
        Needs Postgres and the embedding API (a few minutes). This is how the
        role-phrase option (retrieval.append_role_phrase in the vocabulary
        file) should be judged before anyone switches it on: the cross-encoder
        reranker is very sensitive to query wording, so "phrase" must not move
        the confidence gate for the worse.

    python -m eval.run_understanding_eval --llm
        Judges the optional LLM classifier fallback (QUERY_LLM_CLASSIFIER_*):
        runs eval/llm_classifier_set.json -- phrasings the rules cannot
        classify, plus off-topic questions that must STAY unknown -- with the
        model off and on, and prints both next to the expected label. Makes
        one real model call per row (needs OPENROUTER_API_KEY).
"""
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.query_understanding import understand_query
from retrieval.query_understanding.vocabulary import get_vocabulary

SET_PATH = os.path.join(os.path.dirname(__file__), "query_understanding_set.json")
LLM_SET_PATH = os.path.join(os.path.dirname(__file__), "llm_classifier_set.json")


def run_classification() -> bool:
    vocab = get_vocabulary()
    cases = json.load(open(SET_PATH, encoding="utf-8"))
    correct = 0
    print(f"{'query':58} {'intent':20} {'api_role':24} amb  ok")
    for case in cases:
        r = understand_query(case["query"], case.get("role", "reseller"), vocab=vocab, enabled=True)
        got = (r.intent.value, r.api_role.value, r.ambiguity)
        want = (case["intent"], case["api_role"], case["ambiguity"])
        norm_ok = all(f.lower() in r.normalized_query.lower() for f in case.get("normalized_contains", []))
        ok = got == want and norm_ok
        correct += ok
        print(f"{case['query'][:56]:58} {got[0]:20} {got[1]:24} {str(got[2])[0]:4} {'OK' if ok else 'MISMATCH (want ' + str(want) + ')'}")
    print(f"\nclassification + normalization: {correct}/{len(cases)}")
    return correct == len(cases)


def run_llm_comparison() -> bool:
    vocab = get_vocabulary()
    cases = json.load(open(LLM_SET_PATH, encoding="utf-8"))
    print(f"\n{'query':56} {'expected':22} {'rules':20} {'llm':22} ms")
    rules_ok = llm_ok = 0
    for case in cases:
        want = (case["intent"], case["api_role"], case["ambiguity"])
        rules = understand_query(case["query"], "reseller", vocab=vocab, enabled=True, llm_enabled=False)
        timings: dict = {}
        llm = understand_query(case["query"], "reseller", vocab=vocab, enabled=True, llm_enabled=True, timings=timings)
        got_rules = (rules.intent.value, rules.api_role.value, rules.ambiguity)
        got_llm = (llm.intent.value, llm.api_role.value, llm.ambiguity)
        rules_ok += got_rules == want
        llm_ok += got_llm == want
        note = "" if got_llm == want else f"   <-- got {got_llm}"
        if any(w.startswith("llm_classifier_") for w in llm.warnings):
            note += f"   [{', '.join(w for w in llm.warnings if w.startswith('llm_classifier_'))}]"
        print(f"{case['query'][:54]:56} {want[0]:22} {got_rules[0]:20} {got_llm[0]:22} {timings.get('llm_classification_ms', 0):.0f}{note}")
    print(f"\nexact match (intent + api_role + ambiguity): rules {rules_ok}/{len(cases)}, rules + LLM fallback {llm_ok}/{len(cases)}")
    return True


def run_retrieval_comparison():
    import psycopg

    import config
    from chat import plan_query
    from eval.run_eval import ALL_GROUPS, GOLDEN_SET_PATH
    from retrieval.pipeline import retrieve
    from retrieval.query_rewrite import rewrite_query

    vocab = get_vocabulary()
    phrase_vocab = dataclasses.replace(vocab, append_role_phrase=True)
    golden = json.load(open(GOLDEN_SET_PATH, encoding="utf-8"))
    labeled = json.load(open(SET_PATH, encoding="utf-8"))

    def plans(query, role="reseller"):
        return {
            "legacy": rewrite_query(query, role).rewritten_query,
            "planned": plan_query(query, role, vocab=vocab)[2],
            "phrase": plan_query(query, role, vocab=phrase_vocab)[2],
        }

    cache: dict[str, object] = {}

    def search(conn, text):
        if text not in cache:  # identical plan text -> identical retrieval; don't pay for it twice
            cache[text] = retrieve(conn, text, ALL_GROUPS, mode="parallel")
        return cache[text]

    with psycopg.connect(config.DATABASE_URL) as conn:
        print("\n== golden set: recall@5 and abstain decision per plan ==")
        arms = ("legacy", "planned", "phrase")
        hits = {a: 0 for a in arms}
        answerable = 0
        differing = 0
        for case in golden:
            texts = plans(case["query"])
            differing += len(set(texts.values())) > 1
            row = {}
            for arm, text in texts.items():
                res = search(conn, text)
                top5 = [c.doc_id for c in res.candidates if not c.is_guaranteed][:5]
                row[arm] = (any(d in top5 for d in case.get("expected_doc_ids", [])), res.gate.abstain)
            if case.get("answerable") and not case.get("guaranteed_only"):
                answerable += 1
                for arm in arms:
                    hits[arm] += row[arm][0]
            flag = "" if len({row[a] for a in arms}) == 1 else "   <-- plans disagree"
            print(f"{case['query'][:60]:62}" + "  ".join(f"{a}: hit={row[a][0]!s:5} abstain={row[a][1]!s:5}" for a in arms) + flag)
        print(f"\nrecall@5 over {answerable} answerable questions: " + ", ".join(f"{a}={hits[a]}" for a in arms))
        print(f"{differing} of {len(golden)} golden questions produced a different retrieval text under some plan")

        print("\n== labeled set: top rerank score / abstain per plan (only where the retrieval text differs) ==")
        for case in labeled:
            texts = plans(case["query"], case.get("role", "reseller"))
            if len(set(texts.values())) == 1:
                continue
            print(f"\n{case['query']}")
            for arm, text in texts.items():
                res = search(conn, text)
                top = res.candidates[0].rerank_score if res.candidates and res.candidates[0].rerank_score is not None else 0.0
                print(f"  {arm:8} score={top:.4f} abstain={res.gate.abstain!s:5} text={text[:90]!r}")


if __name__ == "__main__":
    ok = run_classification()
    if "--llm" in sys.argv:
        run_llm_comparison()
    if "--retrieval" in sys.argv:
        run_retrieval_comparison()
    sys.exit(0 if ok else 1)
