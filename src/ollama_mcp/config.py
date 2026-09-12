"""Per-project configuration, loaded from `.ollama-mcp.toml` in the workspace root.

Everything has a working default, so a project with no config file still functions;
the config file exists to name the *verification gate*, which is the only part we
cannot infer safely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - trivial
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py310
    import tomli as tomllib  # type: ignore[no-redef]

CONFIG_NAME = ".ollama-mcp.toml"

DEFAULT_FAST_MODEL = "qwen2.5-coder:7b"
DEFAULT_DEEP_MODEL = "qwen3-coder:30b"

# Ollama's own default is 4096 and it *silently truncates* past that. Never rely on it.
DEFAULT_FAST_NUM_CTX = 16384
DEFAULT_DEEP_NUM_CTX = 32768

DEFAULT_DENY = [
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".ollama-mcp",
    ".env",
    ".env.local",
    "id_rsa",
    "id_ed25519",
    ".pem",
    ".p12",
    "secrets.json",
    "credentials.json",
]


@dataclass
class Models:
    fast: str = DEFAULT_FAST_MODEL
    deep: str = DEFAULT_DEEP_MODEL
    fast_num_ctx: int = DEFAULT_FAST_NUM_CTX
    deep_num_ctx: int = DEFAULT_DEEP_NUM_CTX
    keep_alive: str = "10m"
    temperature: float = 0.1


@dataclass
class Gate:
    """Commands run in the workspace root after an edit. Non-zero exit == failure."""

    commands: list[list[str]] = field(default_factory=list)
    timeout_s: int = 180
    autodetect: bool = True  # fall back to language-native syntax checks


@dataclass
class Limits:
    max_iterations: int = 12
    request_timeout_s: int = 900
    max_receipt_chars: int = 4000
    max_file_bytes: int = 400_000
    max_local_retries: int = 1


@dataclass
class Sandbox:
    deny: list[str] = field(default_factory=lambda: list(DEFAULT_DENY))
    allow_write_outside_git: bool = False


@dataclass
class Config:
    workspace: Path
    models: Models = field(default_factory=Models)
    gate: Gate = field(default_factory=Gate)
    limits: Limits = field(default_factory=Limits)
    sandbox: Sandbox = field(default_factory=Sandbox)
    ollama_host: str = "http://127.0.0.1:11434"
    source: str = "defaults"

    @property
    def state_dir(self) -> Path:
        return self.workspace / ".ollama-mcp"


def _merge(dc: Any, table: dict[str, Any]) -> None:
    for key, value in table.items():
        if hasattr(dc, key):
            setattr(dc, key, value)


def load(workspace: str | Path) -> Config:
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"workspace_root is not a directory: {root}")

    cfg = Config(workspace=root)
    cfg.ollama_host = os.environ.get("OLLAMA_HOST_URL") or _normalize_host(
        os.environ.get("OLLAMA_HOST", cfg.ollama_host)
    )

    path = root / CONFIG_NAME
    if path.is_file():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        _merge(cfg.models, data.get("models", {}))
        _merge(cfg.gate, data.get("gate", {}))
        _merge(cfg.limits, data.get("limits", {}))
        _merge(cfg.sandbox, data.get("sandbox", {}))
        if "ollama_host" in data:
            cfg.ollama_host = _normalize_host(data["ollama_host"])
        cfg.source = str(path)

    # Env always wins, so a user can retarget models without touching the repo.
    cfg.models.fast = os.environ.get("OLLAMA_MCP_FAST_MODEL", cfg.models.fast)
    cfg.models.deep = os.environ.get("OLLAMA_MCP_DEEP_MODEL", cfg.models.deep)
    return cfg


def _normalize_host(host: str) -> str:
    host = host.strip()
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")
