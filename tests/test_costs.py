import pytest

from anvil import costs


def test_cost_math_matches_published_prices():
    assert costs.cost_usd("claude-sonnet-5", 1_000_000, 0) == 3.00
    assert costs.cost_usd("claude-sonnet-5", 0, 1_000_000) == 15.00
    assert costs.cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000) == 6.00
    assert costs.cost_usd("claude-sonnet-5", 1200, 300) == pytest.approx(0.0081)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        costs.cost_usd("gpt-nonexistent", 100, 100)


def test_provider_qualified_and_longest_prefix_pricing():
    # Leading "provider:" prefix is stripped before matching.
    assert costs.cost_usd("anthropic:claude-sonnet-5", 1_000_000, 0) == 3.00
    assert costs.cost_usd("openai:gpt-5.4", 0, 1_000_000) == 15.00
    # Longest matching key wins: gpt-5.4-mini must not match the gpt-5.4 entry.
    assert costs.cost_usd("openai:gpt-5.4-mini", 1_000_000, 0) == 0.75
    assert costs.cost_usd("google_genai:gemini-2.5-flash-lite", 0, 1_000_000) == 0.40


def test_record_degrades_gracefully_for_unpriced_model():
    usd = costs.record("t-x", "answer", "openai:gpt-nonexistent", 1000, 200)
    assert usd == 0.0
    entry = costs.load_ledger()[-1]
    assert entry["cost_usd"] == 0.0
    assert "note" in entry


def test_ledger_report_aggregates_per_conversation():
    costs.record("t1", "router", "claude-haiku-4-5", 500, 10)
    costs.record("t1", "answer", "claude-sonnet-5", 2000, 400)
    costs.record("t2", "answer", "claude-sonnet-5", 1000, 200)
    rep = costs.report()
    assert rep["conversations"] == 2
    assert rep["llm_calls"] == 3
    t1 = rep["threads"]["t1"]
    assert t1["input_tokens"] == 2500
    expected_t1 = (500 * 1 + 10 * 5) / 1e6 + (2000 * 3 + 400 * 15) / 1e6
    assert t1["cost_usd"] == pytest.approx(expected_t1)
    assert rep["mean_cost_per_conversation_usd"] > 0


def test_empty_ledger():
    assert costs.report()["conversations"] == 0
