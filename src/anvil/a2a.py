"""Optional A2A surface for anvil: an Agent Card + an A2A-style task endpoint.

This is a new, separate entrypoint (`anvil-a2a`) that does NOT touch the CLI or MCP server.
It requires the `a2a` extra (`uv sync --extra a2a`). Importing this module for the pure
`build_agent_card` helper needs no extra deps; FastAPI/uvicorn/OTel are imported lazily.

When FLUX_OTLP_ENDPOINT is set, each request joins the caller's distributed trace (by
extracting the incoming W3C traceparent) and emits retrieve/rerank/answer spans carrying
gen_ai token-usage attributes.
"""

import os
from contextlib import nullcontext

from anvil import config

A2A_VERSION = "0.1.0"
SKILL_ID = "answer_grounded_question"


def build_agent_card(base_url: str) -> dict:
    """A real A2A Agent Card (spec 1.0) describing anvil's one skill. Pure; no heavy deps."""
    base = base_url.rstrip("/")
    return {
        "protocolVersion": "1.0",
        "name": "anvil",
        "description": (
            "Support agent over the FastAPI documentation. Answers questions strictly from a "
            "real docs corpus using hybrid retrieval (Postgres FTS + pgvector) and a "
            "cross-encoder reranker, and returns the answer with citations."
        ),
        "url": f"{base}/a2a/tasks",
        "version": A2A_VERSION,
        "provider": {"organization": "zaakirio", "url": "https://github.com/zaakirio"},
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": SKILL_ID,
                "name": "Answer grounded question",
                "description": (
                    "Given a question, retrieves and reranks the most relevant FastAPI "
                    "documentation chunks and returns a grounded answer with citations, or "
                    "declines if the docs do not cover it."
                ),
                "tags": ["rag", "retrieval", "fastapi", "grounded", "citations"],
                "examples": [
                    "How do I enable CORS with CORSMiddleware?",
                    "How do I write tests for my path operations?",
                    "How do I declare dependencies with Depends?",
                ],
            }
        ],
    }


def _synthesize_answer(question: str, chunks: list) -> tuple[str, int, int, str]:
    """Returns (answer_text, input_tokens, output_tokens, model).

    When the configured answer provider's API key is set this calls the real answer
    model (any LangChain-supported provider); keyless it returns a grounded stub over
    the real retrieved citations, with synthetic-but-structured token counts derived
    from the real context size.
    """
    from anvil import providers

    citations = [c.id for c in chunks[:3]]
    context_chars = sum(len(c.content) for c in chunks)

    if providers.provider_key_is_set(config.ANSWER_MODEL):
        from langchain_core.messages import HumanMessage, SystemMessage

        from anvil.agent.nodes import ANSWER_SYSTEM

        context = "\n\n".join(f"[{c.id}] ({c.heading_path})\n{c.content}" for c in chunks)
        model = providers.build_chat_model(config.ANSWER_MODEL, max_tokens=1024)
        resp = model.invoke(
            [SystemMessage(ANSWER_SYSTEM.format(context=context)), HumanMessage(question)]
        )
        usage = getattr(resp, "usage_metadata", None) or {}
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0), config.ANSWER_MODEL

    provider, _ = providers.parse_model_spec(config.ANSWER_MODEL)
    env = providers.key_env_var(provider) or f"{provider.upper()}_API_KEY"
    text = (
        "Based on the FastAPI documentation, the most relevant guidance is in "
        + ", ".join(f"[{cid}]" for cid in citations)
        + f". (Model output is stubbed because no {env} is set; the retrieval, "
        "reranking, and citations above are real.)"
    )
    input_tokens = context_chars // 4 + len(question) // 4
    output_tokens = max(1, len(text) // 4)
    return text, input_tokens, output_tokens, config.ANSWER_MODEL


def answer_grounded_question(question: str, embedder_kind: str = "auto", tracer=None) -> dict:
    """Run anvil's retrieve -> rerank -> answer pipeline, emitting one span per stage."""
    from anvil.rerank import get_reranker
    from anvil.retrieval import search

    def span(name: str):
        return tracer.start_as_current_span(name) if tracer is not None else nullcontext()

    with span("retrieve") as retrieve_span:
        pool = search(question, top_k=config.RERANK_POOL, embedder_kind=embedder_kind, rerank=False)
        if retrieve_span is not None:
            retrieve_span.set_attribute("retrieval.embedder", embedder_kind)
            retrieve_span.set_attribute("retrieval.candidates", len(pool))

    with span("rerank") as rerank_span:
        if pool:
            scores = get_reranker().score(question, [c.content for c in pool])
            for chunk, score in zip(pool, scores, strict=True):
                chunk.rerank_score = score
            pool.sort(key=lambda c: -(c.rerank_score or -1e9))
        top = pool[: config.DEFAULT_TOP_K]
        if rerank_span is not None:
            rerank_span.set_attribute("rerank.model", config.RERANKER_MODEL)
            rerank_span.set_attribute("rerank.pool_size", len(pool))
            rerank_span.set_attribute("rerank.kept", len(top))

    with span("answer") as answer_span:
        text, input_tokens, output_tokens, model = _synthesize_answer(question, top)
        if answer_span is not None:
            from anvil import providers

            answer_span.set_attribute("gen_ai.provider.name", providers.parse_model_spec(model)[0])
            answer_span.set_attribute("gen_ai.request.model", model)
            answer_span.set_attribute("gen_ai.response.model", model)
            answer_span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            answer_span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

    return {
        "answer": text,
        "citations": [{"id": c.id, "heading_path": c.heading_path} for c in top],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "model": model},
    }


def create_app():
    from fastapi import FastAPI, Request

    from anvil import flux_tracing

    base_url = os.environ.get("ANVIL_A2A_BASE_URL", "http://127.0.0.1:8790")
    embedder_kind = os.environ.get("ANVIL_A2A_EMBEDDER", "auto")
    tracer = flux_tracing.setup_tracing("anvil") if flux_tracing.tracing_enabled() else None

    app = FastAPI(title="anvil A2A", version=A2A_VERSION)

    @app.get("/.well-known/agent-card.json")
    def agent_card() -> dict:
        return build_agent_card(base_url)

    @app.post("/a2a/tasks")
    async def run_task(request: Request) -> dict:
        body = await request.json()
        question = body.get("question") or body.get("input") or ""

        if tracer is not None:
            from opentelemetry import context as otel_context

            token = otel_context.attach(flux_tracing.extract_context(dict(request.headers)))
            try:
                with tracer.start_as_current_span("anvil.answer_grounded_question") as root:
                    root.set_attribute("a2a.skill", SKILL_ID)
                    result = answer_grounded_question(question, embedder_kind, tracer)
            finally:
                otel_context.detach(token)
        else:
            result = answer_grounded_question(question, embedder_kind, None)

        return {"skill": SKILL_ID, **result}

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("ANVIL_A2A_HOST", "127.0.0.1")
    port = int(os.environ.get("ANVIL_A2A_PORT", "8790"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
