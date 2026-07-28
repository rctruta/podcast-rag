"""Which model can we actually use right now?

Answers from the environment rather than hardcoding a provider. A visitor
supplies whatever they have — or runs Ollama locally and pays nothing.
"""
from __future__ import annotations

import os
import urllib.request

# litellm model prefix -> the env var that unlocks it
PROVIDERS = {
    "OPENAI_API_KEY":    [("gpt-4o-mini", "OpenAI"), ("gpt-4o", "OpenAI")],
    "ANTHROPIC_API_KEY": [("anthropic/claude-haiku-4-5-20251001", "Anthropic")],
    "GEMINI_API_KEY":    [("gemini/gemini-2.5-flash", "Google")],
    "GROQ_API_KEY":      [("groq/llama-3.3-70b-versatile", "Groq")],
    "OPENROUTER_API_KEY": [("openrouter/meta-llama/llama-3.3-70b-instruct", "OpenRouter")],
}


def ollama_models(host: str = "http://localhost:11434") -> list[tuple[str, str]]:
    """Local models need no key and cost nothing."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as r:
            import json
            data = json.load(r)
        return [(f"ollama/{m['name']}", "Ollama (local, free)")
                for m in data.get("models", [])]
    except Exception:
        return []


def available() -> list[tuple[str, str]]:
    """[(litellm_model_id, provider_label)] for everything usable right now."""
    out: list[tuple[str, str]] = []
    for env, models in PROVIDERS.items():
        if os.environ.get(env):
            out.extend(models)
    out.extend(ollama_models())
    return out


def any_available() -> bool:
    return bool(available())
