"""LLM judge for end-to-end agent evals, following Ragas conventions:

- faithfulness: fraction of claims in the answer supported by the retrieved contexts
- relevancy: how directly the answer addresses the user's question

Both are scored 0.0-1.0 by a Claude judge with a rationale. The judge model is
injectable so the harness is testable keyless with a scripted fake.
"""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from anvil import config

JUDGE_SYSTEM = (
    "You are an evaluation judge for a documentation-grounded support agent. Score the agent's "
    "answer on two axes, each 0.0 to 1.0:\n"
    "- faithfulness: what fraction of the factual claims in the answer are directly supported by "
    "the provided context excerpts? Fabricated or unsupported claims lower the score. An answer "
    "with no factual claims (for example an honest refusal) scores 1.0.\n"
    "- relevancy: how directly and completely does the answer address the user's question? "
    "Evasive, partial, or off-topic answers score low.\n"
    "Reply with ONLY a JSON object: "
    '{"faithfulness": <float>, "relevancy": <float>, "rationale": "<one or two sentences>"}'
)


def _default_judge():
    from anvil.providers import build_chat_model

    return build_chat_model(config.JUDGE_MODEL, max_tokens=512)


def judge_answer(question: str, answer: str, contexts: list[str], judge_model=None) -> dict:
    judge_model = judge_model or _default_judge()
    context_block = "\n\n---\n\n".join(contexts) if contexts else "(no context was retrieved)"
    prompt = (
        f"Question:\n{question}\n\n"
        f"Context excerpts:\n{context_block}\n\n"
        f"Agent answer:\n{answer}"
    )
    resp = judge_model.invoke([SystemMessage(JUDGE_SYSTEM), HumanMessage(prompt)])
    return parse_judge_response(resp.content)


def parse_judge_response(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"judge returned no JSON object: {text!r}")
    data = json.loads(m.group(0))
    return {
        "faithfulness": max(0.0, min(1.0, float(data["faithfulness"]))),
        "relevancy": max(0.0, min(1.0, float(data["relevancy"]))),
        "rationale": str(data.get("rationale", "")),
    }
