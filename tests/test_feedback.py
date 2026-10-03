"""The /api/feedback endpoint: the only source of real-usage signal in this
app (an eval set only tells you about questions someone already thought of).
Logs to its own file, correlated by request_id with that answer's own line
in requests.jsonl -- see server.py's FEEDBACK_LOG_PATH comment."""
import json

import pytest
from fastapi.testclient import TestClient

from app.web import server

VALID_ID = "abcdef012345"  # 12 hex chars, the exact shape uuid.uuid4().hex[:12] produces


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "FEEDBACK_LOG_PATH", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setattr(server, "LOGS_DIR", str(tmp_path))
    return TestClient(server.app)


def read_lines(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def test_a_thumbs_up_is_logged(client):
    resp = client.post("/api/feedback", json={"request_id": VALID_ID, "rating": "up"})
    assert resp.status_code == 200 and resp.json() == {"ok": True}
    lines = read_lines(server.FEEDBACK_LOG_PATH)
    assert len(lines) == 1
    assert lines[0]["request_id"] == VALID_ID and lines[0]["rating"] == "up" and lines[0]["comment"] is None
    assert "ts" in lines[0]


def test_a_thumbs_down_with_a_comment_is_logged(client):
    client.post("/api/feedback", json={"request_id": VALID_ID, "rating": "down", "comment": "wrong API listed"})
    line = read_lines(server.FEEDBACK_LOG_PATH)[0]
    assert line["rating"] == "down" and line["comment"] == "wrong API listed"


def test_a_blank_comment_is_stored_as_none_not_an_empty_string(client):
    client.post("/api/feedback", json={"request_id": VALID_ID, "rating": "down", "comment": "   "})
    assert read_lines(server.FEEDBACK_LOG_PATH)[0]["comment"] is None


def test_an_overlong_comment_is_truncated_not_rejected(client):
    client.post("/api/feedback", json={"request_id": VALID_ID, "rating": "down", "comment": "x" * 5000})
    assert len(read_lines(server.FEEDBACK_LOG_PATH)[0]["comment"]) == server.MAX_FEEDBACK_COMMENT_CHARS


@pytest.mark.parametrize("rating", ["sideways", "UP", "1", ""])
def test_an_invalid_rating_is_rejected(client, rating):
    resp = client.post("/api/feedback", json={"request_id": VALID_ID, "rating": rating})
    assert resp.status_code == 422
    assert not __import__("os").path.exists(server.FEEDBACK_LOG_PATH)


@pytest.mark.parametrize("bad_id", ["", "not-hex-!!", "abc", "a" * 13, "a" * 11, "ABCDEF012345"])
def test_a_malformed_request_id_is_rejected(client, bad_id):
    # request_id is server-generated (uuid4().hex[:12]) and used to join two log
    # files -- reject anything that isn't that exact shape rather than write an
    # arbitrary client-supplied string into a file used for correlation.
    resp = client.post("/api/feedback", json={"request_id": bad_id, "rating": "up"})
    assert resp.status_code == 422


def test_repeated_feedback_for_the_same_request_is_appended_not_overwritten(client):
    # the user can react, then add a comment afterward, or change their mind --
    # every event is kept, so nothing is silently lost or replaced.
    client.post("/api/feedback", json={"request_id": VALID_ID, "rating": "down"})
    client.post("/api/feedback", json={"request_id": VALID_ID, "rating": "down", "comment": "actually, X was missing"})
    lines = read_lines(server.FEEDBACK_LOG_PATH)
    assert len(lines) == 2 and lines[0]["comment"] is None and lines[1]["comment"] == "actually, X was missing"


def test_missing_fields_are_a_422_not_a_500(client):
    resp = client.post("/api/feedback", json={"rating": "up"})
    assert resp.status_code == 422
