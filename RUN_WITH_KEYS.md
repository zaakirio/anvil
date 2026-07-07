# Running the Full Eval Suite With API Keys

Everything below is wired and tested keyless via recorded fixtures.
Once keys exist, these are the exact commands for the real runs.

## 0. Prerequisites (once)

```bash
docker compose up -d postgres
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

## 1. Re-ingest with the primary embedder

The committed baseline was measured with the keyless fallback embedder.
For the primary-embedder numbers, re-ingest so index and queries both use `text-embedding-3-small`:

```bash
uv run anvil ingest --embedder openai
```

## 2. Retrieval eval with OpenAI embeddings

```bash
uv run anvil eval retrieval --embedder openai
```

This writes `eval_results/retrieval-<ts>.json` labeled `openai/text-embedding-3-small` over the same 52 real-question goldens.
Expect a lift over the fallback numbers (recall@5 0.5673, MRR 0.5658); how much is exactly what this run measures.
To make it the new baseline for CI-external comparisons:

```bash
cp "$(ls -t eval_results/retrieval-*.json | head -1)" baselines/retrieval_openai_baseline.json
uv run anvil gate --baseline baselines/retrieval_openai_baseline.json
```

The CI gate stays on the fallback baseline (`baselines/retrieval_baseline.json`) because CI is keyless; do not overwrite it with an OpenAI-embedder run (the gate will refuse the cross-embedder comparison anyway).

## 3. End-to-end agent eval (the real run)

42 golden conversations through the live graph with `claude-sonnet-5` (answer/act) and `claude-haiku-4-5` (router/smalltalk), judged by `claude-sonnet-5`:

```bash
uv run anvil eval agent
```

Note the tool cases fetch live GitHub issues over the public API (10 lookups plus context for 5 comment drafts), so this run needs network access; unauthenticated GitHub API quota (60 requests/hour) is sufficient.
Deterministic checks only (no judge spend):

```bash
uv run anvil eval agent --skip-judge
```

Expected spend: 42 conversations, roughly 60-70 LLM calls plus 26 judge calls; on July 2026 list prices this is on the order of a few tens of cents.
The exact number comes from the ledger, not an estimate:

```bash
uv run anvil report
```

Freeze the first passing run as the agent baseline and gate future runs:

```bash
cp "$(ls -t eval_results/agent-*.json | head -1)" baselines/agent_baseline.json
uv run anvil gate --baseline baselines/agent_baseline.json
```

## 4. Talk to the agent

```bash
uv run anvil ask "How do I enable CORS in FastAPI?"
uv run anvil ask "Look up fastapi issue 2595 and draft a comment explaining the httpx requirement."
uv run anvil chat
```

`anvil chat` pauses inline for comment-draft approvals (the HITL interrupt) and persists threads in a SQLite checkpointer under `.anvil/`.
Approved drafts are recorded in `data/backend/comment_drafts.json`; nothing is ever posted to GitHub.

## 5. Traces in Langfuse (optional)

```bash
docker compose --profile observability up -d
```

Open http://localhost:3000, create an org/project, copy the keys, then:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=http://localhost:3000
```

Every `anvil ask`, `anvil chat`, and `anvil eval agent` run is then traced end to end (router, retrieval, answer, tools, judge).
Unset the keys and tracing is a no-op.
