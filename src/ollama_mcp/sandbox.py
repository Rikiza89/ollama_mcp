"""Path confinement for the local model's tool belt.

The local model writes to the real working tree, so every path it names is
resolved and checked against the workspace root before any I/O happens.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config


class SandboxError(PermissionError):
    """Raised when a path escapes the workspace or hits a deny rule."""


def resolve(cfg: Config, raw: str, *, must_exist: bool = False) -> Path:
    if not raw or not raw.strip():
        raise SandboxError("empty path")

    candidate = Path(raw.strip())
    root = cfg.workspace

    # Absolute paths are allowed only if already inside the workspace.
    target = candidate if candidate.is_absolute() else root / candidate

    try:
        resolved = target.resolve()
    except OSError as exc:  # pragma: no cover - platform dependent
        raise SandboxError(f"cannot resolve path: {raw} ({exc})") from exc

    if resolved != root and root not in resolved.parents:
        raise SandboxError(f"path escapes workspace root: {raw}")

    rel_parts = resolved.relative_to(root).parts
    for token in cfg.sandbox.deny:
        tok = token.lower()
        # Extension-style rule (".pem") or dotted dir/file name (".env").
        if tok.startswith(".") and (
            resolved.name.lower() == tok or resolved.suffix.lower() == tok
        ):
            raise SandboxError(f"denied by sandbox rule {token!r}: {raw}")
        if any(part.lower() == tok for part in rel_parts):
            raise SandboxError(f"denied by sandbox rule {token!r}: {raw}")

    if must_exist and not resolved.exists():
        raise SandboxError(f"no such file: {raw}")
    return resolved


def rel(cfg: Config, path: Path) -> str:
    try:
        return path.resolve().relative_to(cfg.workspace).as_posix()
    except ValueError:  # pragma: no cover - defensive
        return path.as_posix()
