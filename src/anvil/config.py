import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_URL = os.environ.get("ANVIL_DATABASE_URL", "postgresql://anvil:anvil@localhost:5433/anvil")

CORPUS_DIR = Path(os.environ.get("ANVIL_CORPUS_DIR", PROJECT_ROOT / "corpus"))
GOLDENS_DIR = Path(os.environ.get("ANVIL_GOLDENS_DIR", PROJECT_ROOT / "goldens"))
BACKEND_DATA_DIR = Path(os.environ.get("ANVIL_BACKEND_DIR", PROJECT_ROOT / "data" / "backend"))
STATE_DIR = Path(os.environ.get("ANVIL_STATE_DIR", PROJECT_ROOT / ".anvil"))
RESULTS_DIR = Path(os.environ.get("ANVIL_RESULTS_DIR", PROJECT_ROOT / "eval_results"))

CHUNK_MAX_CHARS = int(os.environ.get("ANVIL_CHUNK_MAX_CHARS", "1800"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("ANVIL_CHUNK_OVERLAP_CHARS", "200"))

CANDIDATES_PER_ARM = int(os.environ.get("ANVIL_CANDIDATES_PER_ARM", "50"))
RRF_K = int(os.environ.get("ANVIL_RRF_K", "60"))
RERANK_POOL = int(os.environ.get("ANVIL_RERANK_POOL", "25"))
DEFAULT_TOP_K = int(os.environ.get("ANVIL_TOP_K", "8"))

# Cross-encoder logit below which the agent refuses instead of answering.
# Measured on the real golden set (52 answerable + 10 known-gap questions): the
# populations overlap; answerable rank-1 scores have median 4.4 with 5/52 below 0,
# known gaps range -9.7 to 4.6 with 4/10 below 0. 0.0 (the reranker's own
# relevance boundary) catches the clearly-off queries cheaply; gaps that still
# retrieve plausible-looking chunks are handled by the prompt-layer refusal.
REFUSAL_SCORE_THRESHOLD = float(os.environ.get("ANVIL_REFUSAL_THRESHOLD", "0.0"))

# Generation models are provider-qualified ("provider:model"); see anvil.providers.
# Default to Anthropic/Claude so keyless demos and CI stay deterministic. Switch
# providers by setting these env vars, e.g. ANVIL_ANSWER_MODEL=openai:gpt-5.4, and
# supplying the matching provider key.
ANSWER_MODEL = os.environ.get("ANVIL_ANSWER_MODEL", "anthropic:claude-sonnet-5")
CHEAP_MODEL = os.environ.get("ANVIL_CHEAP_MODEL", "anthropic:claude-haiku-4-5")
JUDGE_MODEL = os.environ.get("ANVIL_JUDGE_MODEL", "anthropic:claude-sonnet-5")

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MAX_STEPS_PER_RUN = int(os.environ.get("ANVIL_MAX_STEPS", "12"))
MAX_TOKENS_PER_RUN = int(os.environ.get("ANVIL_MAX_RUN_TOKENS", "30000"))


def ensure_dirs() -> None:
    for d in (BACKEND_DATA_DIR, STATE_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
