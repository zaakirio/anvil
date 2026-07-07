"""Proves the end-to-end agent eval harness works keyless: the graph runs against
recorded model outputs (fixtures/) and the judge replays recorded verdicts."""

import json
from pathlib import Path

import pytest
from fakes import FakeSearch, ScriptedChatModel, ai
from langgraph.checkpoint.memory import MemorySaver

import anvil.agent.nodes as nodes_mod
from anvil.agent.graph import build_graph
from anvil.agent.nodes import ModelBundle
from anvil.evals.agent import cited_docs, is_refusal, run_agent_eval

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(autouse=True)
def fake_search(monkeypatch):
    monkeypatch.setattr(nodes_mod, "search", FakeSearch())


def load_scripts():
    raw = json.loads((FIXTURES / "agent_eval_scripts.json").read_text())

    def to_msgs(entries):
        return [ai(e["content"], tool_calls=e.get("tool_calls")) for e in entries]

    return to_msgs(raw["cheap_model"]), to_msgs(raw["answer_model"]), to_msgs(raw["judge"])


def test_harness_end_to_end_with_recorded_fixtures(tmp_path):
    cheap, answer, judge_msgs = load_scripts()
    bundle = ModelBundle(
        answer_model=ScriptedChatModel(script=answer),
        cheap_model=ScriptedChatModel(script=cheap),
    )
    graph = build_graph(bundle, checkpointer=MemorySaver())
    judge = ScriptedChatModel(script=judge_msgs)

    result = run_agent_eval(
        graph=graph,
        judge_model=judge,
        goldens_path=FIXTURES / "agent_eval_cases.jsonl",
        out_path=tmp_path / "agent-run.json",
    )

    assert result["cases"] == 3
    assert result["metrics"]["deterministic_pass_rate"] == 1.0
    assert result["metrics"]["faithfulness_mean"] == 1.0
    assert result["metrics"]["relevancy_mean"] == pytest.approx(0.925)

    by_id = {o["id"]: o for o in result["outcomes"]}
    grounded = by_id["fx-grounded"]
    assert grounded["checks"]["citation_present"]
    assert grounded["checks"]["cites_expected_docs"]
    assert grounded["judge"]["faithfulness"] == 1.0

    refusal = by_id["fx-refusal"]
    assert refusal["checks"]["refused_when_should"]

    hitl = by_id["fx-hitl"]
    assert hitl["interrupted"]
    assert hitl["checks"]["hitl_interrupt"]
    assert hitl["checks"]["correct_tool"]
    assert hitl["checks"]["correct_args"]

    written = json.loads((tmp_path / "agent-run.json").read_text())
    assert written["metrics"] == result["metrics"]


def test_refusal_detector():
    assert is_refusal("I could not find an answer to that in the FastAPI documentation.")
    assert is_refusal("The docs do not cover this topic.")
    assert not is_refusal(
        "Add CORSMiddleware with your allowed origins [tutorial-cors#use-corsmiddleware]."
    )


def test_citation_extractor():
    text = "See [tutorial-cors#use-corsmiddleware] and [tutorial-response-model#response-model-priority/2]."
    assert cited_docs(text) == {"tutorial-cors", "tutorial-response-model"}
    assert cited_docs("no citations here") == set()
