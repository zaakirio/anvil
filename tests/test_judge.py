import pytest
from fakes import ScriptedChatModel, ai

from anvil.evals.judge import judge_answer, parse_judge_response


def test_parse_judge_response_plain_json():
    verdict = parse_judge_response('{"faithfulness": 0.9, "relevancy": 0.8, "rationale": "ok"}')
    assert verdict == {"faithfulness": 0.9, "relevancy": 0.8, "rationale": "ok"}


def test_parse_judge_response_with_wrapping_text_and_clamping():
    verdict = parse_judge_response('Sure: {"faithfulness": 1.4, "relevancy": -0.2, "rationale": "x"}')
    assert verdict["faithfulness"] == 1.0
    assert verdict["relevancy"] == 0.0


def test_parse_judge_response_rejects_garbage():
    with pytest.raises(ValueError):
        parse_judge_response("no json here")


def test_judge_answer_with_scripted_model():
    judge = ScriptedChatModel(script=[ai('{"faithfulness": 1.0, "relevancy": 0.9, "rationale": "grounded"}')])
    verdict = judge_answer("q", "a", ["ctx"], judge_model=judge)
    assert verdict["faithfulness"] == 1.0
    assert verdict["rationale"] == "grounded"
