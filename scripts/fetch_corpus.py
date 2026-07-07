"""Fetch and process the real FastAPI documentation into corpus/.

Downloads github.com/fastapi/fastapi at a pinned commit (recorded in
corpus/PROVENANCE.json), selects the tutorial / advanced / deployment / how-to /
reference pages, resolves the docs' code-include directives against docs_src/,
converts MkDocs-Material markup to plain markdown, and writes one flat .md per
page. The processed snapshot is committed so evals are reproducible without
network access; re-run this script only to bump the pinned commit.

Usage: uv run python scripts/fetch_corpus.py
"""

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "corpus"
CACHE_DIR = PROJECT_ROOT / ".cache" / "fastapi"

REPO_URL = "https://github.com/fastapi/fastapi.git"
PINNED_COMMIT = "7cb06f360dd44efac059848df1a9beee7643b018"
LICENSE = "MIT"
DOCS_ROOT = "docs/en/docs"

TOP_LEVEL_PAGES = [
    "index.md",
    "features.md",
    "python-types.md",
    "async.md",
    "fastapi-cli.md",
    "environment-variables.md",
    "virtual-environments.md",
]

REFERENCE_PAGES = [
    "reference/index.md",
    "reference/fastapi.md",
    "reference/testclient.md",
    "reference/uploadfile.md",
    "reference/exceptions.md",
]

GLOB_SECTIONS = ["tutorial/**/*.md", "advanced/**/*.md", "deployment/*.md", "how-to/*.md"]

INCLUDE_STAR_RE = re.compile(r"^\{\*\s+(\S+)\s*([^*]*?)\*\}\s*$")
INCLUDE_BANG_RE = re.compile(r"^\{!>?\s*(\S+?)\s*!\}\s*$")
HEADING_ANCHOR_RE = re.compile(r"^(#{1,6}\s+.*?)\s*\{\s*#[^}]*\}\s*$")
ADMONITION_OPEN_RE = re.compile(r"^///\s*(\w+)(?:\s*\|\s*(.*))?$")
TAB_OPEN_RE = re.compile(r"^////\s*tab\s*\|\s*(.*)$")

FENCE_LANG = {".py": "python", ".js": "javascript", ".html": "html", ".css": "css"}


def checkout_pinned(cache: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(cache), *args], check=True, capture_output=True)

    if not (cache / ".git").exists():
        cache.mkdir(parents=True, exist_ok=True)
        git("init", "-q")
        git("remote", "add", "origin", REPO_URL)
    git("fetch", "--depth", "1", "origin", PINNED_COMMIT)
    git("checkout", "-q", PINNED_COMMIT)


def resolve_source_path(repo: Path, raw: str) -> Path:
    # Include paths are written relative to docs/en (e.g. ../../docs_src/x.py,
    # ../../fastapi/openapi/docs.py); both resolve against the repo root.
    while raw.startswith("../"):
        raw = raw[3:]
    return repo / raw


def slice_lines(text: str, spec: str) -> str:
    # spec like "1:24" or "1:2,12:16,29"; 1-based inclusive ranges.
    lines = text.splitlines()
    keep: list[str] = []
    for part in spec.split(","):
        a, _, b = part.partition(":")
        start, end = int(a), int(b) if b else int(a)
        keep.extend(lines[start - 1 : end])
    return "\n".join(keep)


def render_include(repo: Path, raw_path: str, opts: str) -> list[str]:
    src = resolve_source_path(repo, raw_path)
    text = src.read_text().rstrip("\n")
    m = re.search(r"ln\[([^\]]+)\]", opts)
    if m:
        text = slice_lines(text, m.group(1))
    lang = FENCE_LANG.get(src.suffix, "")
    return [f"```{lang}", *text.splitlines(), "```"]


def process_page(repo: Path, rel_path: str) -> str:
    lines = (repo / DOCS_ROOT / rel_path).read_text().splitlines()
    out: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            m = INCLUDE_BANG_RE.match(stripped)
            if m:
                src = resolve_source_path(repo, m.group(1))
                out.extend(src.read_text().rstrip("\n").splitlines())
                continue
            out.append(line)
            continue
        m = INCLUDE_STAR_RE.match(stripped)
        if m:
            out.extend(render_include(repo, m.group(1), m.group(2)))
            continue
        m = HEADING_ANCHOR_RE.match(line)
        if m:
            out.append(m.group(1))
            continue
        m = TAB_OPEN_RE.match(stripped)
        if m:
            out.append(f"**{m.group(1)}**")
            continue
        if stripped == "////" or stripped == "///":
            continue
        m = ADMONITION_OPEN_RE.match(stripped)
        if m:
            kind = m.group(1).capitalize()
            title = m.group(2)
            out.append(f"**{kind}{': ' + title if title else ''}**")
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n"


def flat_name(rel_path: str) -> str:
    return rel_path.removesuffix(".md").replace("/", "-") + ".md"


def main() -> None:
    checkout_pinned(CACHE_DIR)
    docs = CACHE_DIR / DOCS_ROOT

    pages = list(TOP_LEVEL_PAGES) + list(REFERENCE_PAGES)
    for pattern in GLOB_SECTIONS:
        pages.extend(sorted(str(p.relative_to(docs)) for p in docs.glob(pattern)))

    CORPUS_DIR.mkdir(exist_ok=True)
    for old in CORPUS_DIR.glob("*.md"):
        old.unlink()

    total_bytes = 0
    for rel in pages:
        processed = process_page(CACHE_DIR, rel)
        out_path = CORPUS_DIR / flat_name(rel)
        out_path.write_text(processed)
        total_bytes += len(processed.encode())

    provenance = {
        "source": "https://github.com/fastapi/fastapi",
        "docs_root": DOCS_ROOT,
        "commit": PINNED_COMMIT,
        "license": LICENSE,
        "fetched_at": datetime.now(UTC).isoformat(),
        "pages": len(pages),
        "total_bytes": total_bytes,
        "page_paths": pages,
    }
    (CORPUS_DIR / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {len(pages)} pages ({total_bytes / 1024:.0f} KiB) to {CORPUS_DIR}")


if __name__ == "__main__":
    sys.exit(main())
