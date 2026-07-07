"""Cost ledger: per-call token counts x model prices -> cost per conversation.

Prices are USD per 1M tokens, current list prices as of July 2026.
Sources:
- claude-sonnet-5: $3.00 input / $15.00 output list price. An introductory $2.00/$10.00
  rate applies through 2026-08-31; the ledger uses list price so numbers stay valid
  after the promo. Source: platform.claude.com/docs/en/pricing (verified 2026-07-07).
- claude-haiku-4-5: $1.00 input / $5.00 output. Same source.
- text-embedding-3-small: $0.02 per 1M tokens. Source: platform.openai.com/docs/pricing
  (verified 2026-07-07).
"""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from anvil import config

PRICES_PER_MTOK = {
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "sentence-transformers/all-MiniLM-L6-v2": {"input": 0.0, "output": 0.0},
}

_LOCK = threading.Lock()


def _ledger_path() -> Path:
    config.ensure_dirs()
    return config.STATE_DIR / "ledger.jsonl"


def price_for(model: str) -> dict:
    for known, prices in PRICES_PER_MTOK.items():
        if model.startswith(known):
            return prices
    raise KeyError(f"no price entry for model '{model}'; add it to costs.PRICES_PER_MTOK")


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = price_for(model)
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


def record(thread_id: str, node: str, model: str, input_tokens: int, output_tokens: int) -> float:
    usd = cost_usd(model, input_tokens, output_tokens)
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "thread_id": thread_id,
        "node": node,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(usd, 8),
    }
    with _LOCK, open(_ledger_path(), "a") as f:
        f.write(json.dumps(entry) + "\n")
    return usd


def load_ledger() -> list[dict]:
    path = _ledger_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report() -> dict:
    entries = load_ledger()
    if not entries:
        return {"conversations": 0, "total_cost_usd": 0.0}
    by_thread: dict[str, dict] = {}
    for e in entries:
        t = by_thread.setdefault(
            e["thread_id"],
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        )
        t["calls"] += 1
        t["input_tokens"] += e["input_tokens"]
        t["output_tokens"] += e["output_tokens"]
        t["cost_usd"] += e["cost_usd"]
    total = sum(t["cost_usd"] for t in by_thread.values())
    per_model: dict[str, float] = {}
    for e in entries:
        per_model[e["model"]] = per_model.get(e["model"], 0.0) + e["cost_usd"]
    return {
        "conversations": len(by_thread),
        "llm_calls": len(entries),
        "total_cost_usd": round(total, 6),
        "mean_cost_per_conversation_usd": round(total / len(by_thread), 6),
        "cost_by_model_usd": {m: round(c, 6) for m, c in sorted(per_model.items())},
        "threads": by_thread,
    }
