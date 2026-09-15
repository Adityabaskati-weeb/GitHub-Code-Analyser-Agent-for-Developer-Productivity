"""Small async adapter for Ollama's documented /api/chat endpoint."""
from __future__ import annotations

import asyncio
from time import perf_counter
from urllib.parse import urlparse

import httpx
from langchain_core.messages import AIMessage


class OllamaRequestError(RuntimeError):
    """A local inference failure carrying the attempt/timing evidence."""

    def __init__(self, message: str, metadata: dict | None = None):
        super().__init__(message)
        self.metadata = dict(metadata or {})


class OllamaChat:
    supports_source_schema = True
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

    async def aget_run_metadata(self):
        """Read local runtime/model identity without making an inference call.

        Ollama exposes the runtime version and model digest through read-only
        endpoints.  Metadata discovery is best-effort so an unavailable
        diagnostic endpoint never prevents a model request from being tried.
        """
        result = {"runtime_version": None, "model_digest": None}
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout, 5), trust_env=False) as client:
                version = await client.get(f"{self.base_url}/api/version")
                version.raise_for_status()
                version_data = version.json()
                if isinstance(version_data, dict):
                    result["runtime_version"] = version_data.get("version")

                tags = await client.get(f"{self.base_url}/api/tags")
                tags.raise_for_status()
                tags_data = tags.json()
                models = tags_data.get("models", []) if isinstance(tags_data, dict) else []
                for item in models:
                    if isinstance(item, dict) and item.get("name") == self.model:
                        result["model_digest"] = item.get("digest")
                        break
        except (httpx.HTTPError, ValueError, TypeError, OSError):
            pass
        return result

    def _metadata(self, started: float, attempts: int, data: dict | None = None,
                  status: int | None = None) -> dict:
        """Build durable per-request timing/termination metadata."""
        data = data if isinstance(data, dict) else {}
        metadata = {
            key: data[key] for key in (
                "model", "created_at", "done", "done_reason", "total_duration",
                "load_duration", "prompt_eval_count", "prompt_eval_duration",
                "eval_count", "eval_duration",
            ) if key in data
        }
        metadata.update({
            "inference_attempts": attempts,
            "retry_count": max(0, attempts - 1),
            "request_elapsed_ms": round((perf_counter() - started) * 1000, 3),
        })
        if status is not None:
            metadata["http_status"] = status
        if "done_reason" in data:
            metadata["termination_reason"] = data["done_reason"]
        elif "done" in data:
            metadata["termination_reason"] = "done" if data["done"] else "incomplete"
        return metadata

    async def ainvoke(self, messages, response_format=None):
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        payload = {
            "model": self.model, "stream": False,
            "messages": [{"role": roles[m.type], "content": m.content} for m in messages],
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1024},
        }
        if response_format is not None:
            payload["format"] = response_format
        started = perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            for attempt in range(self.retries + 1):
                attempts = attempt + 1
                try:
                    response = await client.post(f"{self.base_url}/api/chat", json=payload)
                    response.raise_for_status()
                    try:
                        data = response.json()
                    except (ValueError, TypeError) as exc:
                        raise OllamaRequestError(
                            "Ollama returned malformed JSON.",
                            self._metadata(started, attempts, status=response.status_code),
                        ) from exc
                    if not isinstance(data, dict):
                        raise OllamaRequestError(
                            "Ollama returned malformed JSON.",
                            self._metadata(started, attempts, status=response.status_code),
                        )
                    message = data.get("message")
                    if not isinstance(message, dict):
                        raise OllamaRequestError(
                            "Ollama returned malformed JSON.",
                            self._metadata(started, attempts, data, response.status_code),
                        )
                    content = message.get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise OllamaRequestError(
                            "Ollama returned no answer text.",
                            self._metadata(started, attempts, data, response.status_code),
                        )
                    return AIMessage(content=content, response_metadata=self._metadata(
                        started, attempts, data, response.status_code))
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    metadata = self._metadata(started, attempts, status=status)
                    if status == 404:
                        raise OllamaRequestError(
                            f"Local model unavailable. Run: ollama pull {self.model}", metadata
                        ) from exc
                    if status < 500 or attempt == self.retries:
                        raise OllamaRequestError(
                            f"Ollama request failed (HTTP {status}).", metadata
                        ) from exc
                except httpx.TransportError as exc:
                    if attempt == self.retries:
                        raise OllamaRequestError(
                            "Cannot reach local Ollama or inference timed out. "
                            "Start Ollama and verify the model with 'ollama list'.",
                            self._metadata(started, attempts),
                        ) from exc
                await asyncio.sleep(0.5 * (attempt + 1))
