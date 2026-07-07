from langgraph.graph import END, START, StateGraph

from anvil import config
from anvil.agent.nodes import ModelBundle, budget_exceeded, make_nodes
from anvil.agent.state import AgentState


def build_graph(models: ModelBundle | None = None, checkpointer=None, embedder_kind: str = "auto"):
    models = models or ModelBundle.default()
    nodes = make_nodes(models, embedder_kind=embedder_kind)

    g = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        lambda s: "budget_stop" if budget_exceeded(s) else s["route"],
        {"question": "retrieve", "action": "act", "smalltalk": "smalltalk",
         "budget_stop": "budget_stop"},
    )
    g.add_edge("retrieve", "answer")
    g.add_conditional_edges(
        "act",
        lambda s: (
            "budget_stop" if budget_exceeded(s)
            else "tools_exec" if getattr(s["messages"][-1], "tool_calls", None)
            else END
        ),
        {"tools_exec": "tools_exec", "budget_stop": "budget_stop", END: END},
    )
    g.add_edge("tools_exec", "act")
    g.add_edge("answer", END)
    g.add_edge("smalltalk", END)
    g.add_edge("budget_stop", END)

    return g.compile(checkpointer=checkpointer)


def default_checkpointer():
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    config.ensure_dirs()
    conn = sqlite3.connect(str(config.STATE_DIR / "checkpoints.sqlite"), check_same_thread=False)
    return SqliteSaver(conn)
