"""Thin async client for the Ollama HTTP API.

Only the three things we need: health, native tool-calling chat, and a hard
timeout. `num_ctx` is always sent explicitly -- Ollama defaults to 4096 and
truncates silently, which is the single most common cause of a local model
returning a confident, wrong edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


class OllamaError(RuntimeError):
    pass


@dataclass
class ChatResult:
    message: dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return (self.message.get("content") or "").strip()

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return list(self.message.get("tool_calls") or [])


class OllamaClient:
    def __init__(self, host: str, timeout_s: int = 900) -> None:
        self.host = host.rstrip("/")
        self._timeout = httpx.Timeout(timeout_s, connect=10.0)

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            try:
                version = (await client.get(f"{self.host}/api/version")).json()
                tags = (await client.get(f"{self.host}/api/tags")).json()
            except httpx.HTTPError as exc:
                raise OllamaError(f"Ollama unreachable at {self.host}: {exc}") from exc
        return {
            "host": self.host,
            "version": version.get("version", "unknown"),
            "models": sorted(m["name"] for m in tags.get("models", [])),
        }

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        num_ctx: int,
        temperature: float = 0.1,
        keep_alive: str = "10m",
        fmt: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"num_ctx": num_ctx, "temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
        if fmt is not None:
            payload["format"] = fmt

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(f"{self.host}/api/chat", json=payload)
            except httpx.TimeoutException as exc:
                raise OllamaError(
                    f"local model timed out after {self._timeout.read}s (model={model})"
                ) from exc
            except httpx.HTTPError as exc:
                raise OllamaError(f"Ollama request failed: {exc}") from exc

        if response.status_code != 200:
            raise OllamaError(f"Ollama HTTP {response.status_code}: {response.text[:400]}")

        data = response.json()
        if "error" in data:
            raise OllamaError(str(data["error"]))

        return ChatResult(
            message=data.get("message", {}) or {},
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            total_duration_ms=int((data.get("total_duration") or 0) / 1_000_000),
            raw=data,
        )
