import re
from dataclasses import dataclass
from pathlib import Path

from anvil import config


@dataclass
class Chunk:
    id: str
    doc_slug: str
    heading_path: str
    content: str


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    paragraphs = [p for p in re.split(r"\n\n+", text) if p.strip()]
    parts: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            parts.append(current)
            current = current[-overlap:] if overlap else ""
        current = f"{current}\n\n{para}".strip() if current else para
    if current:
        parts.append(current)
    return parts or [text]


def chunk_markdown(
    path: Path,
    max_chars: int = config.CHUNK_MAX_CHARS,
    overlap: int = config.CHUNK_OVERLAP_CHARS,
) -> tuple[str, str, list[Chunk]]:
    """Split one markdown doc into heading-scoped chunks.

    Returns (doc_slug, title, chunks). Chunk ids are stable:
    "<doc-slug>#<heading-slug>" plus "/N" when a section splits.
    """
    doc_slug = path.stem
    lines = path.read_text().splitlines()

    title = doc_slug
    sections: list[tuple[list[str], list[str]]] = []  # (heading trail, body lines)
    trail: list[str] = []
    body: list[str] = []

    def flush():
        if body and any(line.strip() for line in body):
            sections.append((list(trail), list(body)))
        body.clear()

    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            body.append(line)
            continue
        m = None if in_fence else re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1:
                title = heading
                flush()
                trail = []
            else:
                flush()
                trail = trail[: level - 2] + [heading]
        else:
            body.append(line)
    flush()

    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for heading_trail, body_lines in sections:
        heading_path = " > ".join([title, *heading_trail])
        heading_slug = slugify(heading_trail[-1]) if heading_trail else "intro"
        base = f"{doc_slug}#{heading_slug}"
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            base = f"{base}-{seen[base]}"
        text = "\n".join(body_lines).strip()
        # Prefix the heading trail so both lexical and vector search see the context.
        prefixed = f"{heading_path}\n\n{text}"
        parts = _split_long(prefixed, max_chars, overlap)
        for i, part in enumerate(parts):
            cid = base if len(parts) == 1 else f"{base}/{i + 1}"
            chunks.append(Chunk(id=cid, doc_slug=doc_slug, heading_path=heading_path, content=part))
    return doc_slug, title, chunks
