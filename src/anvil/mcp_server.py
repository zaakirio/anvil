"""MCP server (stdio) exposing the same backend the agent uses, so any MCP client
(Claude Desktop, an IDE, another agent) can drive Anvil's knowledge base and actions.

Run: uv run anvil-mcp
"""

import json

from mcp.server.fastmcp import FastMCP

from anvil import backend

mcp = FastMCP("anvil")


@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> str:
    """Hybrid search (full-text + vector + rerank) over the ingested FastAPI documentation.
    Returns the top matching chunks with ids usable as citations."""
    from anvil.retrieval import search

    chunks = search(query, top_k=top_k)
    return json.dumps(
        [
            {
                "id": c.id,
                "heading_path": c.heading_path,
                "rerank_score": c.rerank_score,
                "content": c.content,
            }
            for c in chunks
        ],
        indent=2,
    )


@mcp.tool()
def get_github_issue(number: int, repo: str = backend.DEFAULT_REPO) -> str:
    """Fetch a GitHub issue (title, state, labels, body) from the public GitHub API."""
    return json.dumps(backend.get_github_issue(number=number, repo=repo), indent=2)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
