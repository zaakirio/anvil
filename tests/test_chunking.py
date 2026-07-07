from anvil.chunking import chunk_markdown, slugify


def test_slugify():
    assert slugify("3D Secure") == "3d-secure"
    assert slugify("Who it is for") == "who-it-is-for"
    assert slugify("TLS 1.1") == "tls-1-1"


def test_heading_chunking(tmp_path):
    doc = tmp_path / "sample-doc.md"
    doc.write_text(
        "# Sample Doc\n\nIntro paragraph.\n\n## First Section\n\nBody one.\n\n"
        "## Second Section\n\nBody two.\n\n### Nested Part\n\nBody three.\n"
    )
    slug, title, chunks = chunk_markdown(doc)
    assert slug == "sample-doc"
    assert title == "Sample Doc"
    ids = [c.id for c in chunks]
    assert ids == [
        "sample-doc#intro",
        "sample-doc#first-section",
        "sample-doc#second-section",
        "sample-doc#nested-part",
    ]
    nested = chunks[-1]
    assert nested.heading_path == "Sample Doc > Second Section > Nested Part"
    assert "Body three." in nested.content
    assert nested.content.startswith("Sample Doc > Second Section > Nested Part")


def test_long_section_splits_with_stable_ids(tmp_path):
    doc = tmp_path / "long-doc.md"
    paragraphs = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(12))
    doc.write_text(f"# Long Doc\n\n## Big Section\n\n{paragraphs}\n")
    _, _, chunks = chunk_markdown(doc, max_chars=800, overlap=100)
    assert len(chunks) > 1
    assert all(c.id.startswith("long-doc#big-section/") for c in chunks)
    assert [c.id.rsplit("/", 1)[1] for c in chunks] == [str(i + 1) for i in range(len(chunks))]
    assert all(len(c.content) <= 800 + 400 for c in chunks)


def test_duplicate_headings_get_distinct_ids(tmp_path):
    doc = tmp_path / "dup-doc.md"
    doc.write_text("# Dup\n\n## Notes\n\nA.\n\n## Notes\n\nB.\n")
    _, _, chunks = chunk_markdown(doc)
    assert [c.id for c in chunks] == ["dup-doc#notes", "dup-doc#notes-2"]
