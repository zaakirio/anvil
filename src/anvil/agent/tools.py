import json

from langchain_core.tools import tool

from anvil import backend


@tool
def get_github_issue(number: int, repo: str = backend.DEFAULT_REPO) -> str:
    """Fetch a GitHub issue (title, state, labels, body) from the public GitHub API.

    Use this when the user asks about a specific issue number, wants to know whether a
    problem is already reported, or asks for the status of a bug report.
    """
    try:
        issue = backend.get_github_issue(number=number, repo=repo)
    except Exception as e:  # noqa: BLE001 - surface fetch failures to the model as data
        return json.dumps({"error": f"could not fetch issue: {e}", "repo": repo, "number": number})
    return json.dumps(issue)


@tool
def draft_issue_comment(issue_number: int, body: str, repo: str = backend.DEFAULT_REPO) -> str:
    """Draft a comment on a GitHub issue. Every draft requires human approval before
    anything would be posted; the draft is held until an approver decides."""
    # Execution is intercepted by the HITL approval node; this body never runs in the graph.
    return json.dumps(backend.draft_issue_comment(issue_number, body, repo))


TOOLS = [get_github_issue, draft_issue_comment]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
HITL_TOOLS = {"draft_issue_comment"}
