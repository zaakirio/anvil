"""Real action backends for the agent.

get_github_issue fetches live issues from the GitHub public API (tests replay
recorded fixtures by patching _fetch_issue_json). draft_issue_comment records a
comment draft that is gated behind human approval; nothing is ever posted to
GitHub from this codebase. The HITL approval boundary is the demonstrated part;
actually posting would be one authenticated call after approval.
"""

import json
import threading
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

from anvil import config

_LOCK = threading.Lock()

DEFAULT_REPO = "fastapi/fastapi"
MAX_BODY_CHARS = 2000


def _path(name: str) -> Path:
    config.ensure_dirs()
    return config.BACKEND_DATA_DIR / name


def _load(name: str, default):
    p = _path(name)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def _save(name: str, data) -> None:
    _path(name).write_text(json.dumps(data, indent=2) + "\n")


def _fetch_issue_json(repo: str, number: int) -> dict:
    url = f"https://api.github.com/repos/{repo}/issues/{number}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def get_github_issue(number: int, repo: str = DEFAULT_REPO) -> dict:
    raw = _fetch_issue_json(repo, number)
    body = raw.get("body") or ""
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + " [truncated]"
    return {
        "repo": repo,
        "number": raw["number"],
        "title": raw["title"],
        "state": raw["state"],
        "html_url": raw["html_url"],
        "author": (raw.get("user") or {}).get("login", ""),
        "labels": [label["name"] for label in raw.get("labels", [])],
        "created_at": raw.get("created_at"),
        "closed_at": raw.get("closed_at"),
        "comments": raw.get("comments", 0),
        "body": body,
    }


def draft_issue_comment(issue_number: int, body: str, repo: str = DEFAULT_REPO) -> dict:
    with _LOCK:
        drafts = _load("comment_drafts.json", [])
        draft = {
            "draft_id": f"cmt_{uuid.uuid4().hex[:8]}",
            "repo": repo,
            "issue_number": issue_number,
            "body": body,
            "status": "pending_approval",
            "created_at": datetime.now(UTC).isoformat(),
        }
        drafts.append(draft)
        _save("comment_drafts.json", drafts)
    return draft


def resolve_comment_draft(draft_id: str, approved: bool, approver_note: str = "") -> dict:
    with _LOCK:
        drafts = _load("comment_drafts.json", [])
        for draft in drafts:
            if draft["draft_id"] == draft_id:
                draft["status"] = "approved" if approved else "rejected"
                draft["approver_note"] = approver_note
                draft["resolved_at"] = datetime.now(UTC).isoformat()
                _save("comment_drafts.json", drafts)
                return draft
    raise KeyError(f"unknown comment draft: {draft_id}")
