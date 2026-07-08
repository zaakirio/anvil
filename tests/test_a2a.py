import pytest

from anvil.a2a import SKILL_ID, build_agent_card


def test_agent_card_shape():
    card = build_agent_card("http://127.0.0.1:8790/")
    assert card["protocolVersion"] == "1.0"
    assert card["name"] == "anvil"
    assert card["url"] == "http://127.0.0.1:8790/a2a/tasks"
    assert card["capabilities"]["streaming"] is False
    assert card["defaultOutputModes"] == ["application/json"]


def test_agent_card_advertises_the_skill():
    card = build_agent_card("http://host")
    skills = card["skills"]
    assert len(skills) == 1
    skill = skills[0]
    assert skill["id"] == SKILL_ID
    assert "rag" in skill["tags"]
    assert skill["examples"]


def test_extract_context_joins_incoming_traceparent():
    pytest.importorskip("opentelemetry", reason="a2a extra not installed")
    from opentelemetry import trace

    from anvil.flux_tracing import extract_context

    trace_id_hex = "11112222333344445555666677778888"
    span_id_hex = "aaaabbbbccccdddd"
    ctx = extract_context({"traceparent": f"00-{trace_id_hex}-{span_id_hex}-01"})

    sc = trace.get_current_span(ctx).get_span_context()
    assert format(sc.trace_id, "032x") == trace_id_hex
    assert format(sc.span_id, "016x") == span_id_hex
