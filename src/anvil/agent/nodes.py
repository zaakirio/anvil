import json
from dataclasses import dataclass

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from anvil import backend, costs
from anvil import config as settings
from anvil.agent.state import AgentState
from anvil.agent.tools import HITL_TOOLS, TOOLS, TOOLS_BY_NAME
from anvil.retrieval import search

REFUSAL_TEXT = (
    "I could not find an answer to that in the FastAPI documentation, so I will not guess. "
    "The docs may genuinely not cover it; I can look up a related GitHub issue if you have a "
    "number, or you can rephrase the question."
)

BUDGET_TEXT = (
    "I have hit this conversation's step or token budget, so I am stopping here. "
    "Please start a new conversation or narrow the request."
)

ROUTER_SYSTEM = (
    "You are the router for a support agent that answers questions from the FastAPI "
    "documentation and works with the project's GitHub issues. Classify the latest user "
    "message into exactly one category:\n"
    "- question: asks how to do something with FastAPI, or about its features, behavior, "
    "deployment, or errors\n"
    "- action: asks to look up a specific GitHub issue, or to draft a comment on an issue\n"
    "- smalltalk: greetings, thanks, chit-chat, or anything that is neither of the above\n"
    "Reply with exactly one word: question, action, or smalltalk."
)

ANSWER_SYSTEM = (
    "You are a support agent for FastAPI users. Answer the user's question using ONLY the "
    "FastAPI documentation excerpts below.\n"
    "Rules:\n"
    "1. Every factual claim must carry a citation in square brackets using the excerpt id, "
    "for example [tutorial-query-params#optional-parameters].\n"
    "2. If the excerpts do not contain the answer, say you cannot find it in the documentation "
    "and do not guess.\n"
    "3. If excerpts conflict, prefer current guidance over sections marked deprecated and say "
    "which page is authoritative.\n"
    "4. Be concise and specific.\n\n"
    "Documentation excerpts:\n{context}"
)

ACT_SYSTEM = (
    "You are a support agent for FastAPI users handling an action request. Use the tools to "
    "fetch GitHub issues and to draft issue comments. Comment drafts always go to a human "
    "approver before anything would be posted; tell the user that. When the action is done, "
    "summarize the result including any ids or URLs returned by the tools."
)

SMALLTALK_SYSTEM = (
    "You are a support agent for FastAPI users. Reply briefly and warmly, then remind the user "
    "you can answer questions from the FastAPI documentation, look up GitHub issues, and draft "
    "issue comments for human approval. Two sentences maximum."
)


@dataclass
class ModelBundle:
    """The LLMs behind each node. Swappable so tests can inject scripted fakes."""

    answer_model: object
    cheap_model: object
    answer_model_id: str = settings.ANSWER_MODEL
    cheap_model_id: str = settings.CHEAP_MODEL

    @classmethod
    def default(cls) -> "ModelBundle":
        from anvil.providers import build_chat_model

        return cls(
            answer_model=build_chat_model(settings.ANSWER_MODEL, max_tokens=2048),
            cheap_model=build_chat_model(settings.CHEAP_MODEL, max_tokens=512),
        )


def _thread_id(config: RunnableConfig) -> str:
    return (config.get("configurable") or {}).get("thread_id", "adhoc")


def _invoke(model, model_id: str, node: str, msgs, config: RunnableConfig) -> AIMessage:
    resp = model.invoke(msgs, config)
    usage = getattr(resp, "usage_metadata", None) or {}
    costs.record(
        thread_id=_thread_id(config),
        node=node,
        model=model_id,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )
    return resp


def _tokens(msg: AIMessage) -> int:
    usage = getattr(msg, "usage_metadata", None) or {}
    return usage.get("total_tokens", 0)


def budget_exceeded(state: AgentState) -> bool:
    return (
        state.get("steps", 0) >= settings.MAX_STEPS_PER_RUN
        or state.get("run_tokens", 0) >= settings.MAX_TOKENS_PER_RUN
    )


def make_nodes(models: ModelBundle, embedder_kind: str = "auto") -> dict:
    def router(state: AgentState, config: RunnableConfig) -> dict:
        resp = _invoke(
            models.cheap_model, models.cheap_model_id, "router",
            [SystemMessage(ROUTER_SYSTEM), state["messages"][-1]], config,
        )
        word = resp.content.strip().lower().split()[0] if resp.content.strip() else "question"
        route = word if word in {"question", "action", "smalltalk"} else "question"
        return {"route": route, "steps": state.get("steps", 0) + 1,
                "run_tokens": state.get("run_tokens", 0) + _tokens(resp)}

    def retrieve(state: AgentState, config: RunnableConfig) -> dict:
        query = state["messages"][-1].content
        chunks = search(query, embedder_kind=embedder_kind)
        confidence = max((c.rerank_score or -100.0) for c in chunks) if chunks else -100.0
        return {
            "retrieved": [
                {"id": c.id, "heading_path": c.heading_path, "content": c.content,
                 "rerank_score": c.rerank_score}
                for c in chunks
            ],
            "confidence": confidence,
            "steps": state.get("steps", 0) + 1,
        }

    def answer(state: AgentState, config: RunnableConfig) -> dict:
        chunks = state.get("retrieved", [])
        if not chunks or state.get("confidence", -100.0) < settings.REFUSAL_SCORE_THRESHOLD:
            return {"messages": [AIMessage(REFUSAL_TEXT)], "steps": state.get("steps", 0) + 1}
        context = "\n\n".join(f"[{c['id']}] ({c['heading_path']})\n{c['content']}" for c in chunks)
        resp = _invoke(
            models.answer_model, models.answer_model_id, "answer",
            [SystemMessage(ANSWER_SYSTEM.format(context=context)), *state["messages"]], config,
        )
        return {"messages": [resp], "steps": state.get("steps", 0) + 1,
                "run_tokens": state.get("run_tokens", 0) + _tokens(resp)}

    def act(state: AgentState, config: RunnableConfig) -> dict:
        model = models.answer_model.bind_tools(TOOLS)
        resp = _invoke(
            model, models.answer_model_id, "act",
            [SystemMessage(ACT_SYSTEM), *state["messages"]], config,
        )
        return {"messages": [resp], "steps": state.get("steps", 0) + 1,
                "run_tokens": state.get("run_tokens", 0) + _tokens(resp)}

    def tools_exec(state: AgentState, config: RunnableConfig) -> dict:
        last = state["messages"][-1]
        results: list[ToolMessage] = []
        # HITL calls first: interrupt() replays the node on resume, so side-effecting
        # tools must run only after every approval decision is resolved.
        decisions: dict[str, dict] = {}
        for call in last.tool_calls:
            if call["name"] in HITL_TOOLS:
                decisions[call["id"]] = interrupt(
                    {"action": call["name"], "args": call["args"],
                     "note": "Issue-comment drafts require human approval. Respond with "
                             "{'approved': bool, 'note': str}."}
                )
        for call in last.tool_calls:
            if call["name"] in HITL_TOOLS:
                decision = decisions[call["id"]] or {}
                if decision.get("approved"):
                    draft = backend.draft_issue_comment(**call["args"])
                    resolved = backend.resolve_comment_draft(
                        draft["draft_id"], approved=True, approver_note=decision.get("note", "")
                    )
                    content = json.dumps(resolved)
                else:
                    content = json.dumps(
                        {"status": "rejected_by_approver", "note": decision.get("note", ""),
                         "args": call["args"]}
                    )
            else:
                content = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            results.append(ToolMessage(content=content, tool_call_id=call["id"]))
        return {"messages": results, "steps": state.get("steps", 0) + 1}

    def smalltalk(state: AgentState, config: RunnableConfig) -> dict:
        resp = _invoke(
            models.cheap_model, models.cheap_model_id, "smalltalk",
            [SystemMessage(SMALLTALK_SYSTEM), state["messages"][-1]], config,
        )
        return {"messages": [resp], "steps": state.get("steps", 0) + 1,
                "run_tokens": state.get("run_tokens", 0) + _tokens(resp)}

    def budget_stop(state: AgentState, config: RunnableConfig) -> dict:
        return {"messages": [AIMessage(BUDGET_TEXT)]}

    return {
        "router": router,
        "retrieve": retrieve,
        "answer": answer,
        "act": act,
        "tools_exec": tools_exec,
        "smalltalk": smalltalk,
        "budget_stop": budget_stop,
    }
