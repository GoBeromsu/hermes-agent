"""Unavailable vision routes use the replacement provider's model and fallback policy."""

import asyncio
import json
from unittest.mock import MagicMock

import httpx
import pytest

from agent import auxiliary_client as aux


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("fallback", ["none", "available", "unavailable"])
def test_unavailable_vision_uses_destination_model(tmp_path, monkeypatch, async_mode, fallback):
    """Exercise real config, routing, async conversion and SDK serialization; mock only HTTP."""
    primary_model = "unavailable-primary-model"
    runtime = {
        "provider": "custom", "model": "auto-vision-model",
        "base_url": "https://auto.example.invalid/v1", "api_key": "test-key",
        "api_mode": "chat_completions",
    }
    configured = {
        "provider": "custom", "model": "configured-vision-model",
        "base_url": "https://fallback.example.invalid/v1", "api_key": "test-key",
        "api_mode": "chat_completions",
    }
    chain = [] if fallback == "none" else [configured] if fallback == "available" else [
        {"provider": "unavailable-provider", "model": "unavailable-fallback-model"},
    ]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({"auxiliary": {"vision": {
        "provider": "openai-codex", "model": primary_model, "fallback_chain": chain,
    }}}), encoding="utf-8")
    requests = []
    clients = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append((str(request.url), payload))
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 0,
            "model": payload["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "description"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    def http_client_kwargs(base_url, *, async_mode=False):
        client_cls = httpx.AsyncClient if async_mode else httpx.Client
        client = client_cls(transport=httpx.MockTransport(respond), trust_env=False)
        clients.append(client)
        return {"http_client": client}

    monkeypatch.setattr(aux, "_openai_http_client_kwargs", http_client_kwargs)
    kwargs = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,dGVzdA=="}},
    ]}], "main_runtime": runtime}

    async def call_async():
        try:
            return await aux.async_call_llm("vision", **kwargs)
        finally:
            for client in clients:
                if isinstance(client, httpx.AsyncClient):
                    await client.aclose()

    try:
        response = asyncio.run(call_async()) if async_mode else aux.call_llm("vision", **kwargs)
    finally:
        for client in clients:
            if isinstance(client, httpx.Client):
                client.close()
    destination = configured if fallback == "available" else runtime
    assert response.choices[0].message.content == "description"
    assert len(requests) == 1
    url, payload = requests[0]
    assert url == destination["base_url"] + "/chat/completions"
    assert payload["model"] == destination["model"]
    assert payload["model"] != primary_model
    assert payload["messages"] == kwargs["messages"]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("provider", ["openai-codex", "anthropic", "auto"])
def test_working_vision_routes_keep_selected_model(monkeypatch, async_mode, provider):
    client = MagicMock()
    primary_model = "explicit-vision-model"
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {})
    monkeypatch.setattr(aux, "_resolve_strict_vision_backend", lambda provider, model=None: (client, model))
    monkeypatch.setattr(aux, "_get_cached_client", lambda provider, model=None, async_mode=False, **kwargs: (client, model))
    monkeypatch.setattr(aux, "_vision_main_provider_client", lambda *args: (client, "auto-default"))
    monkeypatch.setattr(aux, "_to_async_client", lambda c, m, **kwargs: (c, m))
    monkeypatch.setattr(aux, "_try_configured_fallback_for_unavailable_client", lambda *args: pytest.fail("working route must not fall back"))
    route = aux._resolve_call_client(
        "vision", provider=provider, model=primary_model, base_url=None, api_key=None,
        resolved_provider=provider, resolved_model=primary_model, resolved_base_url=None,
        resolved_api_key=None, resolved_api_mode=None,
        main_runtime={"provider": "anthropic", "model": "auto-default"}, async_mode=async_mode,
    )
    assert route.client is client
    assert route.final_model == primary_model
    assert route.effective_provider == ("anthropic" if provider == "auto" else provider)
