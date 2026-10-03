"""eval/triage_feedback.py: joining logs/feedback.jsonl with logs/requests.jsonl
by request_id, and the report it builds from that join."""
import json

import pytest

from eval import triage_feedback as triage


@pytest.fixture(autouse=True)
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(triage, "FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setattr(triage, "REQUESTS_PATH", str(tmp_path / "requests.jsonl"))
    return tmp_path


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_no_files_yet_means_no_feedback(logs):
    joined, orphaned = triage.load_feedback_with_context()
    assert joined == [] and orphaned == 0


def test_feedback_is_joined_to_its_request_by_id(logs):
    write_jsonl(triage.REQUESTS_PATH, [{"request_id": "abc123", "query": "which apis can I use", "role": "reseller"}])
    write_jsonl(triage.FEEDBACK_PATH, [{"request_id": "abc123", "rating": "down", "comment": "wrong API", "ts": 1.0}])
    joined, orphaned = triage.load_feedback_with_context()
    assert orphaned == 0
    assert joined[0]["request"]["query"] == "which apis can I use"
    assert joined[0]["comment"] == "wrong API"


def test_feedback_with_no_matching_request_is_counted_as_orphaned_not_dropped(logs):
    write_jsonl(triage.FEEDBACK_PATH, [{"request_id": "ghost", "rating": "up", "ts": 1.0}])
    joined, orphaned = triage.load_feedback_with_context()
    assert orphaned == 1 and len(joined) == 1 and joined[0]["request"] is None


def test_newest_feedback_comes_first(logs):
    write_jsonl(triage.FEEDBACK_PATH, [
        {"request_id": "a", "rating": "up", "ts": 1.0},
        {"request_id": "b", "rating": "down", "ts": 3.0},
        {"request_id": "c", "rating": "up", "ts": 2.0},
    ])
    joined, _ = triage.load_feedback_with_context()
    assert [e["request_id"] for e in joined] == ["b", "c", "a"]


def test_a_malformed_line_is_skipped_not_fatal(logs, capsys):
    with open(triage.FEEDBACK_PATH, "w") as fh:
        fh.write('{"request_id": "a", "rating": "up", "ts": 1.0}\n')
        fh.write("not json at all\n")
    joined, _ = triage.load_feedback_with_context()
    assert len(joined) == 1
    assert "malformed" in capsys.readouterr().err


def test_down_only_filters_to_negative_ratings(logs, capsys):
    write_jsonl(triage.FEEDBACK_PATH, [
        {"request_id": "a", "rating": "up", "ts": 1.0},
        {"request_id": "b", "rating": "down", "ts": 2.0},
    ])
    triage.run_triage(down_only=True)
    out = capsys.readouterr().out
    assert "request_id=b" in out and "request_id=a" not in out


def test_report_surfaces_the_query_and_key_flags(logs, capsys):
    write_jsonl(triage.REQUESTS_PATH, [{
        "request_id": "abc123", "query": "which apis can a distributor use in philipines", "role": "distributor",
        "partner": "acme", "abstained": True, "intent": "unknown", "timings_ms": {"total_ms": 1234.5},
    }])
    write_jsonl(triage.FEEDBACK_PATH, [{"request_id": "abc123", "rating": "down", "comment": "typo wasn't fixed", "ts": 1.0}])
    triage.run_triage()
    out = capsys.readouterr().out
    assert "which apis can a distributor use in philipines" in out
    assert "typo wasn't fixed" in out and "abstained=True" in out and "total_ms=1234.5" in out


def test_no_feedback_at_all_is_reported_plainly(logs, capsys):
    triage.run_triage()
    assert "No feedback logged yet." in capsys.readouterr().out
