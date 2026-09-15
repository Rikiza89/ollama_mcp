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

from . import encoding, i18n

try:  # pragma: no cover - trivial
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py310
    import tomli as tomllib  # type: ignore[no-redef]

class ConfigError(ValueError):
    """A `.ollama-mcp.toml` that cannot be trusted.

    Raised at load time rather than at first use: a typo in the config file
    should stop the server with the file and key named, not silently fall back
    to a default three delegations later.
    """


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
class I18n:
    """Language of the prompts sent to the local model and of the receipt prose.

    `auto` decides per call from the task text, falling back to the machine
    locale. Status tokens (APPLIED, ESCALATE, PASS, FAIL) are protocol and are
    never translated -- see `i18n.py`.
    """

    language: str = i18n.AUTO


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
    i18n: I18n = field(default_factory=I18n)
    ollama_host: str = "http://127.0.0.1:11434"
    source: str = "defaults"

    @property
    def state_dir(self) -> Path:
        return self.workspace / ".ollama-mcp"


def _merge(dc: Any, table: dict[str, Any], *, section: str, source: Path) -> None:
    """Apply one TOML table onto its dataclass, rejecting anything unexpected.

    Raises:
        ConfigError: on an unknown key or a value of the wrong type. `type(...) is`
            rather than `isinstance` on purpose -- `isinstance(True, int)` is True,
            and `max_iterations = true` is not a configuration anyone meant.
    """
    for key, value in table.items():
        if not hasattr(dc, key):
            raise ConfigError(f"{source}: [{section}] has no option {key!r}")
        expected = type(getattr(dc, key))
        if expected is float and type(value) is int:
            value = float(value)
        if type(value) is not expected:
            raise ConfigError(
                f"{source}: [{section}] {key} must be {expected.__name__}, "
                f"got {type(value).__name__}"
            )
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
        # utf-8-sig, not utf-8: a Japanese Windows editor writes a BOM by default
        # and tomllib rejects it as a parse error on line 1, column 1.
        try:
            data = tomllib.loads(encoding.strip_bom(path.read_bytes()).decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"{path}: cannot parse: {exc}") from exc
        for section, target in (
            ("models", cfg.models),
            ("gate", cfg.gate),
            ("limits", cfg.limits),
            ("sandbox", cfg.sandbox),
            ("i18n", cfg.i18n),
        ):
            _merge(target, data.get(section, {}), section=section, source=path)
        if "ollama_host" in data:
            cfg.ollama_host = _normalize_host(data["ollama_host"])
        cfg.source = str(path)

    _validate_gate_commands(cfg.gate, source=cfg.source)

    # Env always wins, so a user can retarget models without touching the repo.
    cfg.models.fast = os.environ.get("OLLAMA_MCP_FAST_MODEL", cfg.models.fast)
    cfg.models.deep = os.environ.get("OLLAMA_MCP_DEEP_MODEL", cfg.models.deep)
    cfg.i18n.language = os.environ.get("OLLAMA_MCP_LANG", cfg.i18n.language)

    try:
        cfg.i18n.language = i18n.normalize_setting(cfg.i18n.language)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{cfg.source}: [i18n] {exc}") from exc
    return cfg


def _validate_gate_commands(gate: Gate, *, source: str) -> None:
    """Check that `commands` really is a list of argv lists.

    `_merge` only sees `type(value) is list` and waves the whole table through,
    which let the obvious TOML mistake -- `commands = ["ruff check ."]`, a list
    of *strings* -- load without complaint. `_run_command` then iterated the
    string one character at a time, reported the gate as `r u f=pass`, and every
    delegated edit afterwards was applied against a gate that checked nothing.

    The gate is the one section the documentation says actually matters, so it
    is worth being strict about, and worth naming the fix in the message.
    """
    for index, command in enumerate(gate.commands):
        where = f"{source}: [gate] commands[{index}]"
        if not isinstance(command, list):
            raise ConfigError(
                f"{where} must be a list of arguments, got {type(command).__name__}. "
                f'Write [["ruff", "check", "."]], not ["ruff check ."].'
            )
        if not command:
            raise ConfigError(f"{where} is empty; a command needs at least a program name")
        bad = next((arg for arg in command if type(arg) is not str), None)
        if bad is not None:
            raise ConfigError(
                f"{where} must contain only strings, got {type(bad).__name__}: {bad!r}"
            )


def _normalize_host(host: str) -> str:
    host = host.strip()
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")
