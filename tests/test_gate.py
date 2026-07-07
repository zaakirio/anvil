import pytest

from anvil.evals.gate import compare


def _run(metrics, embedder="fallback"):
    return {"kind": "retrieval", "embedder": embedder, "metrics": metrics}


BASE = {"recall_at_5": 0.90, "recall_at_10": 0.95, "mrr": 0.85, "ndcg_at_10": 0.88}


def test_gate_passes_within_tolerance():
    run = _run({**BASE, "recall_at_5": 0.89})
    verdict = compare(run, _run(BASE), tolerance=0.02)
    assert verdict["passed"]


def test_gate_fails_on_regression():
    run = _run({**BASE, "mrr": 0.80})
    verdict = compare(run, _run(BASE), tolerance=0.02)
    assert not verdict["passed"]
    failing = [r for r in verdict["rows"] if not r["ok"]]
    assert failing[0]["metric"] == "mrr"


def test_gate_rejects_embedder_mismatch():
    with pytest.raises(ValueError, match="like for like"):
        compare(_run(BASE, embedder="openai/text-embedding-3-small"), _run(BASE), 0.02)


def test_gate_rejects_kind_mismatch():
    agent_run = {"kind": "agent", "metrics": {"deterministic_pass_rate": 1.0}}
    with pytest.raises(ValueError, match="'agent' run against a 'retrieval' baseline"):
        compare(agent_run, _run(BASE), 0.02)


def test_gate_refuses_to_pass_with_no_shared_metrics():
    with pytest.raises(ValueError, match="vacuously"):
        compare(_run({"recall_at_5": 0.9}), _run({"mrr": 0.9}), 0.02)


def test_agent_gate_metrics():
    base = {"kind": "agent", "metrics": {"deterministic_pass_rate": 0.9,
                                         "faithfulness_mean": 0.9, "relevancy_mean": 0.9}}
    run = {"kind": "agent", "metrics": {"deterministic_pass_rate": 0.85,
                                        "faithfulness_mean": 0.92, "relevancy_mean": 0.9}}
    verdict = compare(run, base, tolerance=0.02)
    assert not verdict["passed"]
