"""CI regression gate: compare an eval run to a committed baseline; nonzero exit on
regression beyond thresholds."""

from pathlib import Path

from anvil import config

DEFAULT_TOLERANCE = 0.02

GATED_METRICS = {
    "retrieval": ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"),
    "agent": ("deterministic_pass_rate", "faithfulness_mean", "relevancy_mean"),
}


def latest_run(kind: str) -> Path:
    runs = sorted(config.RESULTS_DIR.glob(f"{kind}-*.json"))
    if not runs:
        raise FileNotFoundError(f"no {kind} eval runs found in {config.RESULTS_DIR}")
    return runs[-1]


def compare(run: dict, baseline: dict, tolerance: float = DEFAULT_TOLERANCE) -> dict:
    kind = baseline.get("kind", "retrieval")
    run_kind = run.get("kind", "retrieval")
    if run_kind != kind:
        raise ValueError(f"cannot gate a '{run_kind}' run against a '{kind}' baseline")
    if baseline.get("embedder") and run.get("embedder") and baseline["embedder"] != run["embedder"]:
        raise ValueError(
            f"baseline embedder '{baseline['embedder']}' != run embedder '{run['embedder']}'; "
            "gate comparisons must be like for like"
        )
    rows = []
    passed = True
    for metric in GATED_METRICS[kind]:
        base = baseline["metrics"].get(metric)
        current = run["metrics"].get(metric)
        if base is None or current is None:
            continue
        delta = round(current - base, 4)
        ok = current >= base - tolerance
        passed = passed and ok
        rows.append({"metric": metric, "baseline": base, "run": current, "delta": delta, "ok": ok})
    if not rows:
        raise ValueError("run and baseline share no gated metrics; the gate cannot pass vacuously")
    return {"kind": kind, "tolerance": tolerance, "passed": passed, "rows": rows}
