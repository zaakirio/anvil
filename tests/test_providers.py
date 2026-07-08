import pytest

from anvil import providers


def test_parse_model_spec_splits_provider_and_model():
    assert providers.parse_model_spec("anthropic:claude-sonnet-5") == ("anthropic", "claude-sonnet-5")
    assert providers.parse_model_spec("openai:gpt-5.4") == ("openai", "gpt-5.4")
    # Only the first colon separates provider from model.
    assert providers.parse_model_spec("google_genai:gemini-2.5-pro") == (
        "google_genai", "gemini-2.5-pro",
    )


def test_parse_model_spec_requires_provider_prefix():
    for bad in ("claude-sonnet-5", "anthropic:", ":gpt-5.4", ""):
        with pytest.raises(ValueError):
            providers.parse_model_spec(bad)


def test_key_env_var_maps_known_providers():
    assert providers.key_env_var("anthropic") == "ANTHROPIC_API_KEY"
    assert providers.key_env_var("openai") == "OPENAI_API_KEY"
    assert providers.key_env_var("google_genai") == "GOOGLE_API_KEY"
    assert providers.key_env_var("nonesuch") is None


def test_provider_key_is_set_reflects_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert providers.provider_key_is_set("anthropic:claude-sonnet-5") is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert providers.provider_key_is_set("anthropic:claude-sonnet-5") is True
    # Unmapped providers are treated as available (auth via other means).
    assert providers.provider_key_is_set("ollama:llama3") is True
