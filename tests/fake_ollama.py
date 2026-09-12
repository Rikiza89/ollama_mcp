"""A scripted stand-in for the Ollama HTTP API.

The real end-to-end path needs a GPU and an 18 GB model, so the test suite runs
against this instead: a real HTTP server on a random port, speaking the same
wire format, replaying a scripted list of assistant turns. That keeps the agent
loop, the tool belt, the gate and the rollback all under test on CI.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


def assistant(content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "model": "fake",
        "message": message,
        "done": True,
        "prompt_eval_count": 100,
        "eval_count": 20,
        "total_duration": 1_000_000,
    }


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"function": {"name": name, "arguments": arguments}}


class FakeOllama:
    """Context manager yielding a base URL. `script` is consumed one turn per chat."""

    def __init__(self, script: list[dict[str, Any]], models: list[str] | None = None) -> None:
        self.script = list(script)
        self.models = models or ["qwen2.5-coder:7b", "qwen3-coder:30b"]
        self.requests: list[dict[str, Any]] = []
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> FakeOllama:  # noqa: PYI034
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence
                pass

            def _send(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/api/version":
                    self._send({"version": "0.0.0-fake"})
                elif self.path == "/api/tags":
                    self._send({"models": [{"name": n} for n in outer.models]})
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(payload)
                if not outer.script:
                    self._send(assistant("DONE: script exhausted"))
                    return
                self._send(outer.script.pop(0))

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)
