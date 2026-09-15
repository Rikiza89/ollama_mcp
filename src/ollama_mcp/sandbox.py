"""Path confinement for the local model's tool belt.

The local model writes to the real working tree, so every path it names is
resolved and checked against the workspace root before any I/O happens.
"""

from __future__ import annotations

import unicodedata
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

    # Containment is checked before this, so the fixup can only ever move between
    # spellings of a path already inside the workspace.
    resolved = _normalized_variant(root, resolved)

    rule = denied_rule(cfg, resolved)
    if rule is not None:
        raise SandboxError(f"denied by sandbox rule {rule!r}: {raw}")

    if must_exist and not resolved.exists():
        raise SandboxError(f"no such file: {raw}")
    return resolved


def denied_rule(cfg: Config, resolved: Path) -> str | None:
    """The first deny rule `resolved` hits, or None if it hits none.

    Public, and the *only* implementation of the rule, because the search tools
    used to carry their own: a part-only, `str.lower()` comparison that never
    looked at extensions. So `read_file("deploy.pem")` was refused while
    `grep` handed the same key back a line at a time, and `list_files` named it.
    A guard with two implementations is a guard with one hole.

    `resolved` is expected to be inside the workspace; anything outside it is
    not this function's business (`resolve` rejects it first) and returns None.
    """
    try:
        parts = resolved.relative_to(cfg.workspace).parts
    except ValueError:
        return None

    for token in cfg.sandbox.deny:
        tok = _fold(token)
        for part in parts:
            folded = _fold(part)
            if folded == tok:
                return token
            # Extension-style rule (".pem", ".env"): any component ending in it.
            # Checked on every component, not just the last, so a file tucked
            # under a denied directory is denied along with it.
            if tok.startswith(".") and folded.endswith(tok):
                return token
    return None


def _fold(text: str) -> str:
    """Comparison key for deny rules: case- and normalization-insensitive.

    Without the NFC step a rule like `"設定"` matches on Linux and silently fails
    on a tree that came from macOS, where the same name is stored decomposed.
    """
    return unicodedata.normalize("NFC", text).lower()


def _normalized_variant(root: Path, resolved: Path) -> Path:
    """Point at an existing file whose name differs only by Unicode normalization.

    macOS stores filenames decomposed (NFD); every editor, every other OS and
    every language model produces composed (NFC). So a repository authored on a
    Mac and checked out on Linux holds files whose names do not byte-match what
    the local model will type, and `読み込み.py` comes back "no such file" while
    sitting right there in the listing.

    Only consulted when the literal path does not exist, so an existing file is
    never re-pointed, and a path for a file about to be *created* keeps exactly
    the spelling that was asked for.
    """
    if resolved == root or resolved.exists():
        return resolved
    relative = resolved.relative_to(root).as_posix()
    for form in ("NFC", "NFD"):
        alternative = unicodedata.normalize(form, relative)
        if alternative != relative and (root / alternative).exists():
            return root / alternative
    return resolved


def rel(cfg: Config, path: Path) -> str:
    try:
        return path.resolve().relative_to(cfg.workspace).as_posix()
    except ValueError:  # pragma: no cover - defensive
        return path.as_posix()
