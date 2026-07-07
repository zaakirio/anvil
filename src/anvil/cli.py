import json
import os
import sys
import uuid
from pathlib import Path

import click

from anvil import config


@click.group()
def cli():
    """Anvil: a support agent over real product docs, with evals as the centerpiece."""


def _require_anthropic_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise click.ClickException("ANTHROPIC_API_KEY is required for this command")


@cli.command()
@click.option("--embedder", "embedder_kind", default="auto",
              type=click.Choice(["auto", "openai", "fallback"]))
@click.option("--corpus", "corpus_dir", default=None, type=click.Path(exists=True, path_type=Path))
def ingest(embedder_kind: str, corpus_dir: Path | None):
    """Chunk and embed the corpus into Postgres (FTS + pgvector)."""
    from anvil.ingest import ingest_corpus

    stats = ingest_corpus(corpus_dir or config.CORPUS_DIR, embedder_kind)
    click.echo(json.dumps(stats, indent=2))


@cli.command()
@click.argument("query")
@click.option("--top-k", default=config.DEFAULT_TOP_K)
@click.option("--embedder", "embedder_kind", default="auto",
              type=click.Choice(["auto", "openai", "fallback"]))
@click.option("--no-rerank", is_flag=True)
def search(query: str, top_k: int, embedder_kind: str, no_rerank: bool):
    """Debug the retrieval pipeline directly."""
    from anvil.retrieval import search as run_search

    for c in run_search(query, top_k=top_k, embedder_kind=embedder_kind, rerank=not no_rerank):
        score = f"{c.rerank_score:.3f}" if c.rerank_score is not None else "-"
        click.echo(f"{score:>8}  [{c.id}]  {c.heading_path}")


def _run_turn(graph, cfg, payload) -> dict:
    from langgraph.types import Command

    result = graph.invoke(payload, cfg)
    while result.get("__interrupt__"):
        intr = result["__interrupt__"][0].value
        click.echo(click.style("\nHUMAN APPROVAL REQUIRED", fg="yellow", bold=True))
        click.echo(json.dumps({"action": intr["action"], "args": intr["args"]}, indent=2))
        approved = click.confirm("Approve this draft?", default=False)
        note = click.prompt("Approver note", default="", show_default=False)
        result = graph.invoke(Command(resume={"approved": approved, "note": note}), cfg)
    return result


def _print_final(result) -> None:
    from langchain_core.messages import AIMessage

    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            click.echo(msg.content if isinstance(msg.content, str) else str(msg.content))
            return


@cli.command()
@click.argument("question")
@click.option("--thread", default=None, help="Conversation thread id (persists across calls).")
def ask(question: str, thread: str | None):
    """One-shot question or action through the full agent graph."""
    _require_anthropic_key()
    from langchain_core.messages import HumanMessage

    from anvil.agent.graph import build_graph, default_checkpointer
    from anvil.observability import get_callbacks

    graph = build_graph(checkpointer=default_checkpointer())
    thread_id = thread or f"cli-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": thread_id}, "callbacks": get_callbacks()}
    result = _run_turn(graph, cfg, {"messages": [HumanMessage(question)]})
    _print_final(result)
    click.echo(click.style(f"\n(thread: {thread_id})", fg="bright_black"))


@cli.command()
def chat():
    """Interactive multi-turn chat with the agent (HITL approvals inline)."""
    _require_anthropic_key()
    from langchain_core.messages import HumanMessage

    from anvil.agent.graph import build_graph, default_checkpointer
    from anvil.observability import get_callbacks

    graph = build_graph(checkpointer=default_checkpointer())
    thread_id = f"chat-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": thread_id}, "callbacks": get_callbacks()}
    click.echo(f"Anvil chat (thread {thread_id}). Ctrl-D or 'exit' to quit.")
    while True:
        try:
            user = click.prompt(click.style("you", fg="cyan"))
        except (EOFError, click.Abort):
            break
        if user.strip().lower() in {"exit", "quit"}:
            break
        result = _run_turn(graph, cfg, {"messages": [HumanMessage(user)]})
        click.echo(click.style("anvil> ", fg="green"), nl=False)
        _print_final(result)


@cli.group(name="eval")
def eval_group():
    """Run the golden-set evals."""


@eval_group.command(name="retrieval")
@click.option("--embedder", "embedder_kind", default="auto",
              type=click.Choice(["auto", "openai", "fallback"]))
@click.option("--no-rerank", is_flag=True)
def eval_retrieval(embedder_kind: str, no_rerank: bool):
    """Retrieval metrics over the golden query set (keyless with --embedder fallback)."""
    from anvil.evals.retrieval import run_retrieval_eval

    result = run_retrieval_eval(embedder_kind=embedder_kind, rerank=not no_rerank)
    click.echo(f"embedder: {result['embedder']}  rerank: {result['rerank']}  "
               f"queries: {result['queries']}")
    for metric, value in result["metrics"].items():
        click.echo(f"  {metric:<14} {value:.4f}")
    click.echo(f"written: {result['out_path']}")


@eval_group.command(name="agent")
@click.option("--skip-judge", is_flag=True, help="Deterministic checks only (no judge calls).")
def eval_agent(skip_judge: bool):
    """End-to-end agent eval over golden conversations. Requires ANTHROPIC_API_KEY."""
    _require_anthropic_key()
    from anvil.evals.agent import run_agent_eval

    result = run_agent_eval(skip_judge=skip_judge)
    click.echo(f"cases: {result['cases']}")
    for metric, value in result["metrics"].items():
        if value is not None:
            click.echo(f"  {metric:<26} {value}")
    click.echo(f"written: {result['out_path']}")


@cli.command()
@click.option("--run", "run_path", default=None, type=click.Path(path_type=Path),
              help="Eval run JSON; defaults to the latest run of the baseline's kind.")
@click.option("--baseline", "baseline_path",
              default=lambda: str(config.PROJECT_ROOT / "baselines" / "retrieval_baseline.json"),
              type=click.Path(exists=True, path_type=Path))
@click.option("--tolerance", default=0.02, show_default=True,
              help="Allowed absolute drop per metric before failing.")
def gate(run_path: Path | None, baseline_path: Path, tolerance: float):
    """Regression gate: compare an eval run against the committed baseline; exit 1 on regression."""
    from anvil.evals.gate import compare, latest_run

    baseline = json.loads(baseline_path.read_text())
    try:
        run_path = run_path or latest_run(baseline.get("kind", "retrieval"))
        verdict = compare(json.loads(run_path.read_text()), baseline, tolerance)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"gate ({verdict['kind']}, tolerance {tolerance}) run={run_path}")
    for row in verdict["rows"]:
        mark = "ok " if row["ok"] else "REG"
        click.echo(f"  {mark} {row['metric']:<26} baseline={row['baseline']:.4f} "
                   f"run={row['run']:.4f} delta={row['delta']:+.4f}")
    if not verdict["passed"]:
        click.echo(click.style("GATE FAILED", fg="red", bold=True))
        sys.exit(1)
    click.echo(click.style("GATE PASSED", fg="green", bold=True))


@cli.command()
def report():
    """Cost report from the per-invocation token ledger."""
    from anvil import costs

    rep = costs.report()
    if not rep.get("conversations"):
        click.echo("ledger is empty; run the agent or the agent eval first")
        return
    click.echo(f"conversations: {rep['conversations']}   llm calls: {rep['llm_calls']}")
    click.echo(f"total cost:    ${rep['total_cost_usd']:.4f}")
    click.echo(f"mean cost per conversation: ${rep['mean_cost_per_conversation_usd']:.4f}")
    click.echo("by model:")
    for model, usd in rep["cost_by_model_usd"].items():
        click.echo(f"  {model:<22} ${usd:.4f}")


@cli.command(name="db-init")
def db_init():
    """Create the Postgres schema (idempotent)."""
    from anvil import db

    db.init_schema()
    click.echo("schema ready")


if __name__ == "__main__":
    cli()
