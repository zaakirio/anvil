import pytest

from anvil.evals.retrieval import _matches, _mrr, _ndcg_at, _recall_at
from anvil.retrieval import _rrf


def test_label_matching_is_exact_or_split_part():
    assert _matches("tutorial-cors#use-corsmiddleware", "tutorial-cors#use-corsmiddleware")
    assert _matches("tutorial-cors#use-corsmiddleware/2", "tutorial-cors#use-corsmiddleware")
    assert not _matches("tutorial-cors#use-corsmiddleware-2", "tutorial-cors#use-corsmiddleware")
    assert not _matches("tutorial-cors#use-corsmiddleware", "tutorial-cors#use-cors")


def test_recall_counts_distinct_labels():
    ranked = ["a#x", "b#y", "c#z"]
    assert _recall_at(ranked, ["a#x", "b#y"], 2) == 1.0
    assert _recall_at(ranked, ["a#x", "missing#m"], 3) == 0.5


def test_mrr_first_relevant_rank():
    assert _mrr(["junk#1", "a#x"], ["a#x"]) == 0.5
    assert _mrr(["junk#1"], ["a#x"]) == 0.0


def test_ndcg_perfect_and_partial():
    assert _ndcg_at(["a#x", "b#y"], ["a#x", "b#y"], 10) == pytest.approx(1.0)
    partial = _ndcg_at(["junk#1", "a#x"], ["a#x"], 10)
    assert 0 < partial < 1


def test_rrf_rewards_agreement():
    fused = _rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] > fused["c"]
    assert fused["b"] > fused["d"]
    assert set(fused) == {"a", "b", "c", "d"}
