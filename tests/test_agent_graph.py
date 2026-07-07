import json
from pathlib import Path

import pytest
from fakes import FakeSearch, ScriptedChatModel, ai
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import anvil.agent.nodes as nodes_mod
from anvil import backend, config
from anvil.agent.graph import build_graph
from anvil.agent.nodes import BUDGET_TEXT, REFUSAL_TEXT, ModelBundle

FIXTURES = Path(__file__).parent.parent / "fixtures" / "github"


@pytest.fixture(autouse=True)
def fake_search(monkeypatch):
    monkeypatch.setattr(nodes_mod, "search", FakeSearch())


@pytest.fixture(autouse=True)
def recorded_issues(monkeypatch):
    def fake_fetch(repo: str, number: int) -> dict:
        return json.loads((FIXTURES / f"issue-{number}.json").read_text())

    monkeypatch.setattr(backend, "_fetch_issue_json", fake_fetch)


def make_graph(cheap_script, answer_script):
    bundle = ModelBundle(
        answer_model=ScriptedChatModel(script=answer_script),
        cheap_model=ScriptedChatModel(script=cheap_script),
    )
    return build_graph(bundle, checkpointer=MemorySaver())


def cfg(tid: str):
    return {"configurable": {"thread_id": tid}}


def test_grounded_question_flow_produces_cited_answer():
    graph = make_graph(
        cheap_script=[ai("question")],
        answer_script=[
            ai("Add CORSMiddleware with your allowed origins [tutorial-cors#use-corsmiddleware].")
        ],
    )
    result = graph.invoke({"messages": [HumanMessage("How can I enable CORS in FastAPI?")]}, cfg("t-q"))
    assert result["route"] == "question"
    assert "[tutorial-cors#use-corsmiddleware]" in result["messages"][-1].content
    assert result["confidence"] > config.REFUSAL_SCORE_THRESHOLD


def test_low_confidence_triggers_deterministic_refusal_without_llm_call():
    answer_model = ScriptedChatModel(script=[])
    bundle = ModelBundle(answer_model=answer_model, cheap_model=ScriptedChatModel(script=[ai("question")]))
    graph = build_graph(bundle, checkpointer=MemorySaver())
    result = graph.invoke(
        {"messages": [HumanMessage("How do I add rate limiting to my endpoints?")]}, cfg("t-r")
    )
    assert result["messages"][-1].content == REFUSAL_TEXT
    assert answer_model.cursor == 0


def test_action_flow_executes_tool_and_summarizes():
    tool_call = {"name": "get_github_issue", "args": {"number": 2026}, "id": "call_1"}
    graph = make_graph(
        cheap_script=[ai("action")],
        answer_script=[ai("", tool_calls=[tool_call]), ai("Issue 2026 is closed.")],
    )
    result = graph.invoke({"messages": [HumanMessage("Look up fastapi issue 2026")]}, cfg("t-a"))
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_msgs) == 1
    issue = json.loads(tool_msgs[0].content)
    assert issue["number"] == 2026
    assert issue["state"] == "closed"
    assert result["messages"][-1].content == "Issue 2026 is closed."


def test_comment_draft_interrupts_and_approval_records_draft():
    tool_call = {"name": "draft_issue_comment",
                 "args": {"issue_number": 2595,
                          "body": "TestClient is now based on httpx; install httpx."},
                 "id": "call_c"}
    graph = make_graph(
        cheap_script=[ai("action")],
        answer_script=[ai("", tool_calls=[tool_call]), ai("Comment draft approved and recorded.")],
    )
    result = graph.invoke(
        {"messages": [HumanMessage("Draft a comment on issue 2595 about httpx")]}, cfg("t-h")
    )
    assert result["__interrupt__"], "expected a HITL interrupt before recording the draft"
    payload = result["__interrupt__"][0].value
    assert payload["action"] == "draft_issue_comment"
    assert payload["args"]["issue_number"] == 2595

    result = graph.invoke(Command(resume={"approved": True, "note": "accurate"}), cfg("t-h"))
    tool_msg = [m for m in result["messages"] if m.type == "tool"][0]
    body = json.loads(tool_msg.content)
    assert body["status"] == "approved"
    drafts = json.loads((config.BACKEND_DATA_DIR / "comment_drafts.json").read_text())
    assert drafts[0]["issue_number"] == 2595


def test_comment_draft_rejection_does_not_touch_backend():
    tool_call = {"name": "draft_issue_comment",
                 "args": {"issue_number": 1663, "body": "Read the docs more carefully."},
                 "id": "call_c2"}
    graph = make_graph(
        cheap_script=[ai("action")],
        answer_script=[ai("", tool_calls=[tool_call]), ai("The approver rejected this draft.")],
    )
    graph.invoke({"messages": [HumanMessage("Draft a dismissive comment on issue 1663")]}, cfg("t-h2"))
    result = graph.invoke(Command(resume={"approved": False, "note": "tone"}), cfg("t-h2"))
    tool_msg = [m for m in result["messages"] if m.type == "tool"][0]
    assert json.loads(tool_msg.content)["status"] == "rejected_by_approver"
    assert not (config.BACKEND_DATA_DIR / "comment_drafts.json").exists()


def test_smalltalk_route():
    graph = make_graph(
        cheap_script=[ai("smalltalk"), ai("Hi! I can answer FastAPI documentation questions.")],
        answer_script=[],
    )
    result = graph.invoke({"messages": [HumanMessage("hello!")]}, cfg("t-s"))
    assert "FastAPI" in result["messages"][-1].content


def test_step_budget_stops_runaway_tool_loop(monkeypatch):
    monkeypatch.setattr(config, "MAX_STEPS_PER_RUN", 3)
    tool_call = {"name": "get_github_issue", "args": {"number": 2026}, "id": "loop"}
    looping = [ai("", tool_calls=[{**tool_call, "id": f"loop{i}"}]) for i in range(10)]
    graph = make_graph(cheap_script=[ai("action")], answer_script=looping)
    result = graph.invoke({"messages": [HumanMessage("loop forever")]}, cfg("t-b"))
    assert result["messages"][-1].content == BUDGET_TEXT


def test_router_records_costs_in_ledger():
    from anvil import costs

    graph = make_graph(
        cheap_script=[ai("smalltalk"), ai("Hello from the FastAPI support agent.")],
        answer_script=[],
    )
    graph.invoke({"messages": [HumanMessage("hey")]}, cfg("t-c"))
    entries = costs.load_ledger()
    assert {e["node"] for e in entries} == {"router", "smalltalk"}
    assert all(e["cost_usd"] > 0 for e in entries)
