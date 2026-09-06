import asyncio
from unittest.mock import patch

import httpx
import pytest
from langchain_core.messages import HumanMessage

from src.llm.ollama import OllamaChat
from src.config.settings import get_llm


def invoke(handler, retries=0):
    original = httpx.AsyncClient
    def client(**kwargs):
        return original(transport=httpx.MockTransport(handler), **kwargs)
    with patch("src.llm.ollama.httpx.AsyncClient", side_effect=client):
        return asyncio.run(OllamaChat("test-model", retries=retries).ainvoke([HumanMessage(content="hello")]))


def test_local_provider_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(get_llm("ollama"), OllamaChat)


def test_request_uses_nonstreaming_chat_and_keeps_usage():
    import json
    def handler(request):
        data = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert data["stream"] is False
        assert data["messages"] == [{"role":"user", "content":"hello"}]
        assert data["model"] == "test-model"
        return httpx.Response(200, json={"message":{"content":"answer"}, "eval_count":3})
    result = invoke(handler)
    assert result.content == "answer"
    assert result.response_metadata["eval_count"] == 3


def test_missing_model_has_actionable_error():
    with pytest.raises(RuntimeError, match="ollama pull test-model"):
        invoke(lambda _: httpx.Response(404))


def test_timeout_has_actionable_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)
    with pytest.raises(RuntimeError, match="timed out"):
        invoke(handler)


def test_transient_server_error_retries_once():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json={"message":{"content":"ok"}})
    assert invoke(handler, retries=1).content == "ok"
    assert len(calls) == 2


def test_client_error_not_retried():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(400)
    with pytest.raises(RuntimeError, match="400"):
        invoke(handler, retries=1)
    assert len(calls) == 1


def test_empty_answer_rejected():
    with pytest.raises(RuntimeError, match="no answer text"):
        invoke(lambda _: httpx.Response(200, json={"message":{"content":""}}))


@pytest.mark.parametrize("url", ["https://example.com", "file:///tmp/model", "http://user:pass@localhost:11434"])
def test_local_mode_rejects_remote_endpoints(url):
    with pytest.raises(ValueError, match="local loopback"):
        OllamaChat("model", url)
