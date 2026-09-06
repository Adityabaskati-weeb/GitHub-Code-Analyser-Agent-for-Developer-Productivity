"""Small async adapter for Ollama's documented /api/chat endpoint."""
from __future__ import annotations

import asyncio
from urllib.parse import urlparse

import httpx
from langchain_core.messages import AIMessage


class OllamaChat:
    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 timeout: float = 120, retries: int = 1):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "localhost", "127.0.0.1", "::1"
        } or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Ollama must use a local loopback HTTP(S) URL.")
        if timeout <= 0 or retries < 0:
            raise ValueError("Timeout must be positive and retries nonnegative.")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.timeout, self.retries = timeout, retries

    async def ainvoke(self, messages):
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        payload = {
            "model": self.model, "stream": False,
            "messages": [{"role": roles[m.type], "content": m.content} for m in messages],
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1024},
        }
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            for attempt in range(self.retries + 1):
                try:
                    response = await client.post(f"{self.base_url}/api/chat", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    content = data.get("message", {}).get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise RuntimeError("Ollama returned no answer text.")
                    return AIMessage(content=content, response_metadata={
                        key: data[key] for key in (
                            "model", "total_duration", "prompt_eval_count", "eval_count"
                        ) if key in data
                    })
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status == 404:
                        raise RuntimeError(
                            f"Local model unavailable. Run: ollama pull {self.model}"
                        ) from exc
                    if status < 500 or attempt == self.retries:
                        raise RuntimeError(f"Ollama request failed (HTTP {status}).") from exc
                except httpx.TransportError as exc:
                    if attempt == self.retries:
                        raise RuntimeError(
                            "Cannot reach local Ollama or inference timed out. "
                            "Start Ollama and verify the model with 'ollama list'."
                        ) from exc
                await asyncio.sleep(0.5 * (attempt + 1))
