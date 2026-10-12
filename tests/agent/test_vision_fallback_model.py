"""Unavailable vision routes must not carry their model into another provider."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from agent import auxiliary_client as aux


@pytest.mark.parametrize("async_mode", [False, True])
def test_unavailable_vision_uses_destination_model(tmp_path, monkeypatch, async_mode):
    primary_model = "gpt-6-astra"
    auto_model = "auto-selected-vision-model"
    fallback_model = "configured-fallback-model"
    runtime = {"provider": "anthropic", "model": auto_model}
    client = MagicMock()
    response = MagicMock(choices=[MagicMock(message=MagicMock(content="description"))])
    client.chat.completions.create = AsyncMock(return_value=response) if async_mode else MagicMock(return_value=response)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(aux, "_resolve_strict_vision_backend", lambda *args: (None, None))
    monkeypatch.setattr(aux, "_get_cached_client", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(aux, "_vision_main_provider_client", lambda *args: (client, auto_model))
    monkeypatch.setattr(aux, "_to_async_client", lambda c, m, **kwargs: (c, m))
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda *args: False)
    fallback_entries = []

    def resolve_fallback(entry):
        fallback_entries.append(entry)
        return client, entry["model"]

    monkeypatch.setattr(aux, "_resolve_fallback_entry", resolve_fallback)
    for chain in ([], [{"provider": "anthropic", "model": fallback_model}]):
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"auxiliary": {"vision": {
            "provider": "openai-codex", "model": primary_model, "fallback_chain": chain,
        }}}))
        expected_model = chain[0]["model"] if chain else aux._vision_auto_route(runtime, None, None, async_mode)[2]
        kwargs = {"messages": [{"role": "user", "content": "describe"}], "main_runtime": runtime}
        result = asyncio.run(aux.async_call_llm("vision", **kwargs)) if async_mode else aux.call_llm("vision", **kwargs)
        assert result is response
        assert client.chat.completions.create.call_args.kwargs["model"] == expected_model
        assert expected_model != primary_model
    assert fallback_entries == chain


@pytest.mark.parametrize("async_mode", [False, True])
def test_working_vision_routes_keep_selected_model(monkeypatch, async_mode):
    client = MagicMock()
    primary_model = "gpt-6-astra"
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {})
    monkeypatch.setattr(aux, "_resolve_strict_vision_backend", lambda provider, model=None: (client, model))
    monkeypatch.setattr(aux, "_get_cached_client", lambda provider, model=None, async_mode=False, **kwargs: (client, model))
    monkeypatch.setattr(aux, "_vision_main_provider_client", lambda *args: (client, "auto-default"))
    monkeypatch.setattr(aux, "_to_async_client", lambda c, m, **kwargs: (c, m))
    monkeypatch.setattr(aux, "_try_configured_fallback_for_unavailable_client", lambda *args: pytest.fail("working route must not fall back"))
    for provider in ("openai-codex", "anthropic", "auto"):
        route = aux._resolve_call_client(
            "vision", provider=provider, model=primary_model, base_url=None, api_key=None,
            resolved_provider=provider, resolved_model=primary_model, resolved_base_url=None,
            resolved_api_key=None, resolved_api_mode=None,
            main_runtime={"provider": "anthropic", "model": "auto-default"}, async_mode=async_mode,
        )
        assert route.client is client
        assert route.final_model == primary_model
        assert route.effective_provider == ("anthropic" if provider == "auto" else provider)
