"""Provider-agnostic resolution for anvil's generation models.

Generation models are configured as provider-qualified specs, e.g.
`anthropic:claude-sonnet-5` or `openai:gpt-5.4`. The provider selects both the
LangChain integration package (via `init_chat_model`) and the API-key env var
used to gate the keyless deterministic path. Anthropic/Claude stays the default
so keyless demos and CI stay deterministic.
"""

import os

# provider -> the env var that carries its API key. Covers the common
# init_chat_model providers; any LangChain-supported provider works so long as
# its integration package is installed and the matching key is set.
PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "google_vertexai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
}


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split a `provider:model` spec into `(provider, model)`."""
    provider, sep, model = spec.partition(":")
    if not sep or not provider or not model:
        raise ValueError(
            f"model spec {spec!r} must be provider-qualified, "
            "e.g. 'anthropic:claude-sonnet-5' or 'openai:gpt-5.4'"
        )
    return provider, model


def key_env_var(provider: str) -> str | None:
    """The API-key env var for a provider, or None if the provider isn't mapped."""
    return PROVIDER_KEY_ENV.get(provider)


def provider_key_is_set(spec: str) -> bool:
    """True if the spec's provider has its API-key env var set to a non-empty value.

    Providers not in PROVIDER_KEY_ENV (authenticated by other means, e.g. Vertex
    application-default credentials) are treated as available.
    """
    provider, _ = parse_model_spec(spec)
    env = key_env_var(provider)
    if env is None:
        return True
    return bool(os.environ.get(env))


def build_chat_model(spec: str, **kwargs):
    """Instantiate a LangChain chat model from a `provider:model` spec.

    `init_chat_model` lazily imports the provider's integration package and raises
    a clear ImportError (with an install hint) if the extra is not installed.
    """
    from langchain.chat_models import init_chat_model

    provider, model = parse_model_spec(spec)
    return init_chat_model(model, model_provider=provider, **kwargs)
