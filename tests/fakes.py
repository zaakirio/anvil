"""Scripted stand-ins for chat models so the agent graph and eval harness run keyless."""

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted AIMessages in order, regardless of input."""

    script: list[AIMessage]
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.cursor >= len(self.script):
            raise AssertionError(f"scripted model exhausted after {len(self.script)} calls")
        msg = self.script[self.cursor]
        object.__setattr__(self, "cursor", self.cursor + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self


def ai(content: str, tool_calls: list[dict] | None = None,
       input_tokens: int = 100, output_tokens: int = 25) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


class FakeSearch:
    """Replaces anvil.agent.nodes.search: high-confidence chunks unless the query
    matches a configured gap marker."""

    def __init__(self, gap_markers: tuple[str, ...] = ("rate limit", "favicon", "mongodb")):
        self.gap_markers = gap_markers

    def __call__(self, query: str, **kwargs):
        from anvil.retrieval import RetrievedChunk

        lowered = query.lower()
        if any(m in lowered for m in self.gap_markers):
            return [
                RetrievedChunk(id="tutorial-middleware#create-a-middleware",
                               doc_slug="tutorial-middleware",
                               heading_path="Middleware > Create a middleware",
                               content="You can add middleware to FastAPI applications.",
                               fused_score=0.01, rerank_score=-8.5)
            ]
        return [
            RetrievedChunk(id="tutorial-cors#use-corsmiddleware", doc_slug="tutorial-cors",
                           heading_path="CORS > Use CORSMiddleware",
                           content="You can configure it in your FastAPI application using the "
                                   "CORSMiddleware: import it and add it with add_middleware, "
                                   "passing the allowed origins.", fused_score=0.03,
                           rerank_score=7.2),
            RetrievedChunk(id="tutorial-cors#intro", doc_slug="tutorial-cors",
                           heading_path="CORS (Cross-Origin Resource Sharing)",
                           content="CORS refers to situations when a frontend running in a "
                                   "browser talks to a backend on a different origin.",
                           fused_score=0.02, rerank_score=5.1),
        ]
