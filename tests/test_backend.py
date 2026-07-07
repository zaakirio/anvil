import json
from pathlib import Path

import pytest

from anvil import backend

FIXTURES = Path(__file__).parent.parent / "fixtures" / "github"


@pytest.fixture
def recorded_issues(monkeypatch):
    def fake_fetch(repo: str, number: int) -> dict:
        path = FIXTURES / f"issue-{number}.json"
        if not path.exists():
            raise FileNotFoundError(f"no recorded fixture for issue {number}")
        return json.loads(path.read_text())

    monkeypatch.setattr(backend, "_fetch_issue_json", fake_fetch)


def test_get_github_issue_from_recorded_fixture(recorded_issues):
    issue = backend.get_github_issue(number=2026)
    assert issue["number"] == 2026
    assert issue["state"] == "closed"
    assert "403" in issue["title"]
    assert issue["html_url"] == "https://github.com/fastapi/fastapi/issues/2026"
    assert issue["repo"] == "fastapi/fastapi"


def test_get_github_issue_truncates_long_bodies(recorded_issues, monkeypatch):
    monkeypatch.setattr(backend, "MAX_BODY_CHARS", 100)
    issue = backend.get_github_issue(number=2595)
    assert issue["body"].endswith("[truncated]")
    assert len(issue["body"]) <= 100 + len(" [truncated]")


def test_comment_draft_and_resolution():
    draft = backend.draft_issue_comment(2595, "TestClient now requires httpx.")
    assert draft["status"] == "pending_approval"
    assert draft["repo"] == "fastapi/fastapi"
    resolved = backend.resolve_comment_draft(draft["draft_id"], approved=True, approver_note="ok")
    assert resolved["status"] == "approved"
    rejected = backend.draft_issue_comment(1663, "Please read the docs.")
    resolved = backend.resolve_comment_draft(rejected["draft_id"], approved=False)
    assert resolved["status"] == "rejected"
    with pytest.raises(KeyError):
        backend.resolve_comment_draft("cmt_missing", approved=True)
