import json

import pytest

import chat
from app.web import server
from retrieval.query_understanding import understand_query


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(server, "LOG_PATH", str(path))
    return path


def make_response(query="which APIs can I implement for biz trade in?", **kw):
    understanding = understand_query(query, "reseller", enabled=True).to_dict()
    return chat.ChatResponse(
        query=query, role="reseller", partner="acme", answer="a", understanding=understanding,
        retrieval_result_count=7, timings_ms={"understand_ms": 1.2, "total_ms": 900.0}, **kw,
    )


def test_log_line_has_the_new_structured_fields(log_path):
    req = server.ChatRequest(query="which APIs can I implement for biz trade in?", role="reseller", partner="acme")
    server._log_request(req, make_response(), 901.0, "abc123")
    line = json.loads(log_path.read_text())
    assert line["request_id"] == "abc123"
    assert (line["intent"], line["api_role"], line["ambiguity"]) == ("api_implementation", "unclear", True)
    assert line["normalization_count"] == 2 and line["corrected_terms_count"] == 0
    assert line["retrieval_query_count"] == 1 and line["retrieval_result_count"] == 7
    assert line["retrieval_fallback_used"] is False and line["understanding_fallback"] is False
    assert line["timings_ms"]["understand_ms"] == 1.2 and line["total_wall_ms"] == 901.0


def test_log_records_no_more_user_text_than_it_did_before(log_path):
    req = server.ChatRequest(query="which APIs can I implement for biz trade in?", role="reseller", partner="acme")
    server._log_request(req, make_response(), 1.0, "id")
    line = json.loads(log_path.read_text())
    # `query` was already logged before query understanding existed; nothing else may carry user text.
    assert "trade" not in json.dumps({k: v for k, v in line.items() if k != "query"})
    for forbidden in ("answer", "normalized_query", "retrieval_query", "clarifying_question", "corrected_terms", "expansion_terms"):
        assert forbidden not in line


def test_refusals_without_an_understanding_still_log(log_path):
    req = server.ChatRequest(query="commission payouts", role="reseller", partner="acme")
    server._log_request(req, chat.ChatResponse(query="commission payouts", role="reseller", partner="acme", answer="no",
                                                permission_refused=True), 2.0, "id")
    line = json.loads(log_path.read_text())
    assert line["intent"] is None and line["permission_refused"] is True


def test_fallback_is_flagged_in_the_log(log_path):
    req = server.ChatRequest(query="q", role="reseller", partner="acme")
    response = chat.ChatResponse(query="q", role="reseller", partner="acme", answer="a",
                                  understanding=understand_query("q", "reseller", enabled=False).to_dict())
    server._log_request(req, response, 1.0, "id")
    assert json.loads(log_path.read_text())["understanding_fallback"] is True


def test_stream_sends_a_correlation_id_first_and_logs_it(log_path, monkeypatch):
    monkeypatch.setattr(server, "prepare_answer", lambda *a, **k: make_response("what apis are available to me"))
    events = list(server._stream_chat(server.ChatRequest(query="what apis are available to me", role="reseller", partner="acme")))
    first = json.loads(events[0].split("data: ")[1])
    assert first["phase"] == "retrieving" and len(first["request_id"]) == 12
    assert events[-1].startswith("event: final")
    assert json.loads(log_path.read_text())["request_id"] == first["request_id"]


def test_final_event_carries_the_understanding_and_clarification_flag(monkeypatch, log_path):
    monkeypatch.setattr(server, "prepare_answer", lambda *a, **k: make_response(clarification=True))
    events = list(server._stream_chat(server.ChatRequest(query="q", role="reseller", partner="acme")))
    final = json.loads(events[-1].split("data: ")[1])
    assert final["clarification"] is True and final["understanding"]["intent"] == "api_implementation"
