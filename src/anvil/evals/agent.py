"""End-to-end agent evals: golden conversations graded by deterministic checks plus an
LLM judge (faithfulness / relevancy per Ragas conventions).

Golden case shape (goldens/agent.jsonl):
{
  "id": "...", "type": "grounded|refusal|tool|hitl|smalltalk", "user": "...",
  "checks": {
    "cite_docs": ["tutorial-cors"],          # each slug must appear in a citation
    "must_refuse": true,                     # refusal expected
    "tool": "get_github_issue",              # expected tool call
    "args_subset": {"number": 2026},         # expected args are a subset of actual
    "hitl": true, "approve": true            # HITL interrupt expected; approval decision
  }
}
Requires model keys for the real run; the harness itself is exercised keyless in tests
via scripted fakes and recorded fixtures.
"""

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from anvil import config
from anvil.agent.nodes import REFUSAL_TEXT
from anvil.evals.judge import judge_answer
from anvil.observability import get_callbacks

CITATION_RE = re.compile(r"\[([a-z0-9-]+)#[a-z0-9/-]+\]")

REFUSAL_MARKERS = (
    "could not find",
    "cannot find",
    "can't find",
    "not cover",
    "n't cover",
    "no information",
    "don't have information",
    "do not have information",
    "unable to find",
    "not documented",
)


def load_goldens(path: Path | None = None) -> list[dict]:
    path = path or config.GOLDENS_DIR / "agent.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_refusal(text: str) -> bool:
    lower = text.lower()
    return REFUSAL_TEXT.lower() in lower or any(m in lower for m in REFUSAL_MARKERS)


def cited_docs(text: str) -> set[str]:
    return set(CITATION_RE.findall(text))


def _final_ai_text(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _tool_calls(messages) -> list[dict]:
    calls = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            calls.append(call)
    return calls


def run_case(graph, case: dict) -> dict:
    thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    cfg = {"configurable": {"thread_id": thread_id}, "callbacks": get_callbacks()}
    checks = case.get("checks", {})

    result = graph.invoke({"messages": [HumanMessage(case["user"])]}, cfg)
    interrupted = bool(result.get("__interrupt__"))
    if interrupted:
        decision = {"approved": bool(checks.get("approve", False)), "note": "golden-set decision"}
        result = graph.invoke(Command(resume=decision), cfg)

    final = _final_ai_text(result["messages"])
    calls = _tool_calls(result["messages"])
    retrieved = result.get("retrieved", [])

    outcome = {
        "id": case["id"],
        "type": case["type"],
        "thread_id": thread_id,
        "final_answer": final,
        "interrupted": interrupted,
        "tool_calls": [{"name": c["name"], "args": c["args"]} for c in calls],
        "checks": {},
    }

    if checks.get("cite_docs"):
        docs = cited_docs(final)
        outcome["checks"]["citation_present"] = bool(docs)
        outcome["checks"]["cites_expected_docs"] = all(d in docs for d in checks["cite_docs"])
    if "must_refuse" in checks:
        outcome["checks"]["refused_when_should"] = is_refusal(final) == checks["must_refuse"]
    if checks.get("tool"):
        matching = [c for c in calls if c["name"] == checks["tool"]]
        outcome["checks"]["correct_tool"] = bool(matching)
        if matching and checks.get("args_subset"):
            args = matching[0]["args"]
            outcome["checks"]["correct_args"] = all(
                args.get(k) == v for k, v in checks["args_subset"].items()
            )
    if "hitl" in checks:
        outcome["checks"]["hitl_interrupt"] = interrupted == checks["hitl"]

    outcome["deterministic_pass"] = all(outcome["checks"].values()) if outcome["checks"] else True
    outcome["contexts"] = [c["content"] for c in retrieved]
    return outcome


def run_agent_eval(
    graph=None,
    judge_model=None,
    goldens_path: Path | None = None,
    out_path: Path | None = None,
    skip_judge: bool = False,
) -> dict:
    if graph is None:
        from anvil.agent.graph import build_graph
        from anvil.agent.nodes import ModelBundle

        graph = build_graph(ModelBundle.default(), checkpointer=MemorySaver())

    goldens = load_goldens(goldens_path)
    outcomes = []
    for case in goldens:
        outcome = run_case(graph, case)
        if not skip_judge and case["type"] in ("grounded", "refusal"):
            outcome["judge"] = judge_answer(
                question=case["user"],
                answer=outcome["final_answer"],
                contexts=outcome["contexts"],
                judge_model=judge_model,
            )
        outcome.pop("contexts", None)
        outcomes.append(outcome)

    judged = [o for o in outcomes if "judge" in o]
    n = len(outcomes)
    metrics = {
        "deterministic_pass_rate": round(sum(o["deterministic_pass"] for o in outcomes) / n, 4),
        "faithfulness_mean": (
            round(sum(o["judge"]["faithfulness"] for o in judged) / len(judged), 4) if judged else None
        ),
        "relevancy_mean": (
            round(sum(o["judge"]["relevancy"] for o in judged) / len(judged), 4) if judged else None
        ),
    }
    result = {
        "kind": "agent",
        "ts": datetime.now(UTC).isoformat(),
        "cases": n,
        "metrics": metrics,
        "outcomes": outcomes,
    }
    config.ensure_dirs()
    out = out_path or config.RESULTS_DIR / f"agent-{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    result["out_path"] = str(out)
    return result
