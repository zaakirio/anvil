"""Cost ledger: per-call token counts x model prices -> cost per conversation.

Prices are USD per 1M tokens, current list prices as of July 2026. Keys are matched
against the bare model name (any leading "provider:" prefix is stripped) by longest
prefix, so provider-qualified ids like "anthropic:claude-sonnet-5" price correctly.
Sources:
- Anthropic: platform.claude.com/docs/en/pricing (verified 2026-07-08).
  - claude-sonnet-5: $3.00 in / $15.00 out list price. An introductory $2.00/$10.00 rate
    applies through 2026-08-31; the ledger uses list price so numbers stay valid after it.
  - claude-haiku-4-5: $1.00 in / $5.00 out.
- OpenAI: developers.openai.com/api/docs/pricing (verified 2026-07-08). The gpt-4.x models
  are retired from the pricing page; the current text tier is gpt-5.x.
  - gpt-5.5: $5.00 in / $30.00 out. gpt-5.4: $2.50 in / $15.00 out.
  - gpt-5.4-mini: $0.75 in / $4.50 out. gpt-5.4-nano: $0.20 in / $1.25 out.
  - text-embedding-3-small: $0.02 in (embeddings, no output tokens).
- Google Gemini: ai.google.dev/gemini-api/docs/pricing, paid tier (verified 2026-07-08).
  - gemini-2.5-pro: $1.25 in / $10.00 out for prompts <=200k tokens (>200k: $2.50/$15.00).
  - gemini-2.5-flash: $0.30 in / $2.50 out. gemini-2.5-flash-lite: $0.10 in / $0.40 out.
  - gemini-2.0-flash: $0.10 in / $0.40 out. gemini-2.0-flash-lite: $0.075 in / $0.30 out.
"""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from anvil import config

PRICES_PER_MTOK = {
    # Anthropic
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # OpenAI
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    # Google Gemini (gemini-2.5-pro is the <=200k-token tier)
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    # Local fallback embedder (runs on CPU, no API cost)
    "sentence-transformers/all-MiniLM-L6-v2": {"input": 0.0, "output": 0.0},
}

_LOCK = threading.Lock()


def _ledger_path() -> Path:
    config.ensure_dirs()
    return config.STATE_DIR / "ledger.jsonl"


def price_for(model: str) -> dict:
    # Strip any leading "provider:" prefix, then match the longest known key.
    bare = model.split(":", 1)[1] if ":" in model else model
    for known in sorted(PRICES_PER_MTOK, key=len, reverse=True):
        if bare.startswith(known):
            return PRICES_PER_MTOK[known]
    raise KeyError(f"no price entry for model '{model}'; add it to costs.PRICES_PER_MTOK")


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = price_for(model)
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


def record(thread_id: str, node: str, model: str, input_tokens: int, output_tokens: int) -> float:
    # An unpriced model must not crash the run; record it at 0 with a note instead.
    try:
        usd = cost_usd(model, input_tokens, output_tokens)
        note = None
    except KeyError:
        usd = 0.0
        note = f"no price entry for model '{model}'; recorded at 0"
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "thread_id": thread_id,
        "node": node,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(usd, 8),
    }
    if note:
        entry["note"] = note
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
