"""Generation-quality eval: unlike run_eval.py (did retrieval find the right
DOCUMENT?) this checks whether the actual ANSWER TEXT is right -- correct
facts present, nothing forbidden claimed, citations attached, and genuinely
unanswerable questions actually get refused. Runs the full pipeline end to
end (real Postgres, real embedding API, real LLM) -- this is the thing a
user actually reads, and nothing else in this repo's eval suite checks it.

    python -m eval.run_generation_eval

Each case in generation_eval_set.json is one of:
  - an answerable question: `expected_facts` (all must appear, case-
    insensitive substring match -- deliberately simple and durable to
    paraphrasing, not exact-string matching) and optionally
    `forbidden_facts` (none may appear -- catches a wrong business-rule
    claim, e.g. code produced for a role that should have been refused)
    and `min_citations`.
  - a genuinely unanswerable question: `must_abstain: true` -- checks the
    answer contains the app's refusal phrase and has no citations. This
    covers BOTH ways a request ends up unanswered: the confidence gate
    abstaining before generation, and the grounding-based refusal the LLM
    itself gives when the retrieved context turns out not to support an
    answer (see llm/generate.py's SYSTEM_PROMPT) -- checking the answer
    text, not just the `abstained` flag, catches both.

Known limitation, not hidden: the LLM is not perfectly deterministic even at
temperature=0 (documented elsewhere in this repo -- the auth-question
refusal flakiness). A case can flip between runs for reasons that have
nothing to do with a real regression. Facts here were chosen to be the
parts of the answer that stayed stable across repeated live runs before
being added -- see each case's git history for what was checked. A single
run of this script is a spot check, not a certainty; a case that fails once
is worth a second run before treating it as a real regression.

This set is meant to GROW from real usage, not stay fixed: a real production
failure, once understood and fixed, belongs here as a permanent case --
see docs/eval-and-feedback.md for the loop this is one half of.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REFUSAL_PHRASE = "I don't have that information"

SET_PATH = os.path.join(os.path.dirname(__file__), "generation_eval_set.json")


def _check(case: dict, response) -> tuple[bool, list[str]]:
    """Returns (passed, problems) -- problems is empty when passed is True,
    otherwise a list of every mismatch found (a case can fail more than one
    way at once, and seeing all of them beats stopping at the first)."""
    problems = []
    answer_lower = response.answer.lower()

    if case.get("must_abstain"):
        if REFUSAL_PHRASE.lower() not in answer_lower:
            problems.append(f"expected a refusal ({REFUSAL_PHRASE!r}) but got: {response.answer[:100]!r}")
        if response.citations:
            problems.append(f"a refusal should carry no citations, got {len(response.citations)}")
        return not problems, problems

    for fact in case.get("expected_facts", []):
        if fact.lower() not in answer_lower:
            problems.append(f"missing expected fact: {fact!r}")
    for fact in case.get("forbidden_facts", []):
        if fact.lower() in answer_lower:
            problems.append(f"contains forbidden fact: {fact!r}")
    min_citations = case.get("min_citations", 0)
    if len(response.citations) < min_citations:
        problems.append(f"only {len(response.citations)} citation(s), expected at least {min_citations}")
    return not problems, problems


def run_generation_eval(mode: str = "parallel") -> bool:
    import chat

    cases = json.load(open(SET_PATH, encoding="utf-8"))
    passed_count = 0
    print(f"{'query':60} {'role':12} result")
    for case in cases:
        response = chat.answer_query(case["query"], case.get("role", "reseller"), case.get("partner", "acme"), mode=mode)
        passed, problems = _check(case, response)
        passed_count += passed
        print(f"{case['query'][:58]:60} {case.get('role', 'reseller'):12} {'PASS' if passed else 'FAIL'}")
        for problem in problems:
            print(f"    - {problem}")
        if not passed:
            print(f"    answer was: {response.answer[:200]!r}")

    print(f"\ngeneration eval: {passed_count}/{len(cases)} passed")
    print(
        "note: a fresh LLM call happens for every case, every run -- a case that fails once, alone,\n"
        "may be run-to-run variance (see this module's docstring) rather than a real regression;\n"
        "re-run before treating a single failure as confirmed."
    )
    return passed_count == len(cases)


if __name__ == "__main__":
    ok = run_generation_eval(sys.argv[1] if len(sys.argv) > 1 else "parallel")
    sys.exit(0 if ok else 1)
