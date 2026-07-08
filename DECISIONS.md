# Design Decisions

## Corpus: real documentation at a pinned commit, snapshot committed

The corpus is the actual FastAPI documentation (MIT licensed), not synthetic pages, because an eval only means something if the retrieval difficulty is real.
`scripts/fetch_corpus.py` pins commit `7cb06f360dd44efac059848df1a9beee7643b018` of `fastapi/fastapi`, selects 118 pages (tutorial, advanced, deployment, how-to, reference, core top-level), inlines the `{* docs_src/... *}` code includes so chunks contain the code users ask about, and flattens MkDocs-Material markup (admonitions, tabs, heading anchors) to plain markdown.
The processed snapshot (~0.9 MB) is committed so `anvil ingest` and the evals are reproducible offline; the script exists to bump the pin, not as a runtime dependency.
Provenance (source, commit, page list, license) is recorded in `corpus/PROVENANCE.json`.

## Golden sets: mined from real user questions, labeled by inspection

Every retrieval golden is a real question: 45 Stack Overflow titles and 7 FastAPI GitHub issues, chosen for being answerable from the ingested pages, each with its source URL recorded in the golden file.
Relevant chunks were labeled by reading the corpus and deciding which sections actually answer the question; that human judgment is the legitimate manual step in golden-set curation, and the per-query eval output exists so any label can be audited.
Refusal goldens are real questions the docs genuinely do not answer (rate limiting, favicons, refresh tokens, API versioning, MongoDB), verified by grepping the corpus before inclusion.
Real questions are messier than synthetic ones (vague titles, error dumps, XY problems), which drops the headline metrics relative to a toy corpus and keeps the eval representative of real support traffic.

## Chunking: heading-aware markdown, not fixed windows

The corpus is documentation with strong heading structure, so chunk boundaries follow headings rather than fixed token windows.
Each `##`/`###` section becomes one chunk, prefixed with its full heading trail ("CORS > Use CORSMiddleware") so both lexical and vector search see the document context.
Fenced code blocks are tracked so a Python `#` comment at column 0 is never mistaken for a heading, which matters in code-heavy docs.
Sections longer than 1,800 characters split on paragraph boundaries with a 200 character overlap, producing ids like `tutorial-response-model#response-model-priority/2`.
This gives stable, human-readable chunk ids (`doc-slug#heading-slug`) that double as citation targets and as golden-set labels, which is what makes the retrieval eval labelable at all.
The tradeoff is uneven chunk sizes; for documentation corpora that costs less than splitting an explanation across two windows.

## Retrieval: hybrid FTS + vector with RRF, then cross-encoder rerank

Postgres full-text search (`tsvector`, `ts_rank_cd`) catches exact terminology like `CORSMiddleware` and `root_path`; pgvector cosine over embeddings catches paraphrases.
Each arm returns 50 candidates and the lists are fused with Reciprocal Rank Fusion (k=60), which needs no score calibration between arms.
The fused top 25 are reranked with the `ms-marco-MiniLM-L-6-v2` cross-encoder on CPU, the standard second-stage reranking setup.
Measured on the real golden set with the fallback embedder, reranking lifts recall@5 from 0.4615 to 0.5673, MRR from 0.4077 to 0.5658, and nDCG@10 from 0.4385 to 0.5847; on real questions the reranker is not a nicety, it is most of the ranking quality.
The rerank score also gives the agent a grounding-confidence signal that pure RRF scores cannot provide.

## Embeddings: OpenAI primary, sentence-transformers fallback for keyless CI

`text-embedding-3-small` is the primary embedder for real deployments.
CI cannot hold API keys hostage to a merge, so the eval gate runs on `all-MiniLM-L6-v2` locally on CPU, clearly labeled in every eval artifact via the `embedder` field.
The gate refuses to compare runs across embedders, and retrieval refuses to query an index built with a different embedder than the query embedder.
The vector column is untyped (`vector` without a fixed dimension) so both 384-dim and 1536-dim indexes work against one schema; at 1,199 chunks an exact scan is faster than maintaining an ANN index, so none is created.

## Refusal: deterministic threshold first, prompt refusal second

The answer node refuses without calling the model when the best rerank score is below a threshold.
Measured on the real golden set, the populations overlap: answerable questions have median rank-1 score 4.4 with 5/52 below 0, and known-gap questions range -9.7 to 4.6 with 4/10 below 0.
The threshold sits at 0.0, the cross-encoder's own relevance boundary: it catches clearly-off queries for free and false-refuses roughly one answerable question in ten at the tail.
Gap questions that still retrieve plausible chunks (a middleware question when middleware pages exist but do not answer it) pass the threshold and rely on the prompt-layer instruction to refuse when the excerpts do not contain the answer.
A clean threshold split was a property of the earlier synthetic corpus, not of real data; the two-layer design is the response to that.

## Tools: real APIs, and a write boundary that never writes upstream

`get_github_issue` calls the public GitHub REST API and trims the response to the fields the model needs; tests replay recorded fixtures (`fixtures/github/`) by patching the fetch function, so the suite runs offline and deterministic.
`draft_issue_comment` is the one action with external consequences, so it is gated with a LangGraph `interrupt()` plus a SQLite checkpointer; issue lookups execute directly.
Interrupts run before any side-effecting tool in the same turn because LangGraph replays the node on resume; executing tools before an unresolved interrupt would double-execute them.
Approved drafts are recorded as approved but never posted: posting needs a GitHub token and a deliberate integration, and the thing this project demonstrates is the approval boundary, not the POST request.

## Evals: two layers, gated separately

Retrieval evals label relevant chunks per query (52 real queries) and compute recall@5, recall@10, MRR, and nDCG@10; they run keyless and gate CI.
Agent evals grade whole conversations (42 cases from the same real-question mining): deterministic checks (citation present, refused-when-should, correct tool and args, HITL interrupt observed) plus an LLM judge scoring faithfulness and relevancy per Ragas conventions.
Deterministic checks are separated from judge scores because the former are free and exact while the latter cost money and carry judge variance; the gate can hold both to different tolerances.
Golden labels are validated against the ingested corpus before every eval run so a renamed heading fails loudly instead of silently zeroing recall.

## Cost ledger: record at the call site, price from a source-commented table

Every LLM call in the graph records its node, model, and token counts to a JSONL ledger keyed by conversation thread.
Prices live in one table in `costs.py` with source comments and verification dates; an unknown model raises instead of silently costing zero.
`anvil report` aggregates to cost per conversation, which is the number a CTO actually asks for.

## Langfuse: self-hosted, env-gated to no-op

Tracing uses Langfuse's LangChain callback, attached only when `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set, so keyless runs and CI carry zero tracing overhead or config burden.
The self-hosted v3 stack (web, worker, Postgres, ClickHouse, Redis, MinIO) ships in `docker-compose.yml` behind the `observability` profile so the core Postgres stays a one-container start.

## Fakes over mocks for the agent tests

The graph accepts a `ModelBundle`, and tests inject scripted chat models that replay recorded outputs (including tool calls and usage metadata) from `fixtures/`.
This exercises the real graph, real routing, real interrupt machinery, and the real eval harness keyless, rather than asserting on mocked call signatures.
