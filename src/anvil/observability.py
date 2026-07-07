"""Langfuse integration, env-gated: no credentials means no-op (empty callback list).

Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST (self-hosted:
http://localhost:3000) and every agent run is traced end to end.
"""

import os


def langfuse_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def get_callbacks() -> list:
    if not langfuse_enabled():
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]
