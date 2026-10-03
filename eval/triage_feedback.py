"""Joins logs/feedback.jsonl (a 👍/👎 + optional comment from the chat UI, see
app/web/server.py's /api/feedback) with logs/requests.jsonl (every request's
own structured log line, keyed by the same request_id) and prints a readable
report -- the practical tool for the loop docs/eval-and-feedback.md
describes: real usage surfaces a failure, a human reads it WITH its full
context in one place, and turns it into either a code fix, a new row in
eval/generation_eval_set.json or eval/golden_set.json, or both. Without the
join, a bare "👎, wrong API listed" tells you nothing actionable; with it,
you also see the exact query, role, intent, retrieval scores, and whether it
abstained -- everything needed to reproduce and fix it.

    python -m eval.triage_feedback                # every rated request, newest first
    python -m eval.triage_feedback --down-only     # just the negative ratings -- the actionable ones
"""
import json
import os
import sys
from collections import defaultdict

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
FEEDBACK_PATH = os.path.join(LOGS_DIR, "feedback.jsonl")
REQUESTS_PATH = os.path.join(LOGS_DIR, "requests.jsonl")


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  (skipping malformed line {lineno} in {path})", file=sys.stderr)
    return rows


def load_feedback_with_context() -> tuple[list[dict], int]:
    """Returns (joined rows newest-first, count of feedback entries with no
    matching request -- e.g. requests.jsonl was rotated/cleared after the
    fact; still worth surfacing rather than silently dropping)."""
    requests_by_id: dict[str, dict] = {}
    for row in _read_jsonl(REQUESTS_PATH):
        requests_by_id[row.get("request_id", "")] = row  # last one wins if a request_id ever repeats

    joined = []
    orphaned = 0
    for fb in _read_jsonl(FEEDBACK_PATH):
        request = requests_by_id.get(fb.get("request_id"))
        if request is None:
            orphaned += 1
        joined.append({**fb, "request": request})
    joined.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return joined, orphaned


def format_entry(entry: dict) -> str:
    badge = {"up": "\U0001F44D", "down": "\U0001F44E"}.get(entry["rating"], entry["rating"])
    lines = [f"{badge}  request_id={entry['request_id']}"]
    if entry.get("comment"):
        lines.append(f"     comment: {entry['comment']!r}")
    req = entry.get("request")
    if req is None:
        lines.append("     (no matching request log line -- requests.jsonl may have rotated)")
    else:
        lines.append(f"     query: {req.get('query', '')!r}  (role={req.get('role')}, partner={req.get('partner')})")
        flags = ", ".join(
            f"{k}={req[k]}" for k in ("intent", "api_role", "abstained", "permission_refused", "classifier") if req.get(k) not in (None, False)
        )
        if flags:
            lines.append(f"     {flags}")
        total_ms = (req.get("timings_ms") or {}).get("total_ms")
        if total_ms is not None:
            lines.append(f"     total_ms={total_ms}")
    return "\n".join(lines)


def run_triage(down_only: bool = False) -> None:
    joined, orphaned = load_feedback_with_context()
    if down_only:
        joined = [e for e in joined if e["rating"] == "down"]

    if not joined:
        print("No feedback logged yet." if not down_only else "No 👎 feedback logged yet.")
        return

    counts = defaultdict(int)
    for e in joined:
        counts[e["rating"]] += 1
    print(f"{len(joined)} feedback entries" + (f" ({orphaned} with no matching request log line)" if orphaned else ""))
    print(f"  up={counts['up']}  down={counts['down']}\n")

    for entry in joined:
        print(format_entry(entry))
        print()

    print(
        "Turning a real failure into a permanent test: add it to eval/generation_eval_set.json\n"
        "(expected_facts / forbidden_facts / must_abstain) or eval/golden_set.json (expected_doc_ids),\n"
        "the same way every fix in this project's history has grown those sets -- see\n"
        "docs/eval-and-feedback.md."
    )


if __name__ == "__main__":
    run_triage(down_only="--down-only" in sys.argv)
