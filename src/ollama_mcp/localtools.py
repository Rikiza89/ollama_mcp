"""The tool belt handed to the *local* model.

These execute on this machine only. Nothing here ever reaches Claude's context --
Claude sees the receipt, not these results. Kept deliberately small: every tool
schema costs the local model context too, and small models degrade fast with a
wide belt.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import encoding
from .config import Config
from .sandbox import SandboxError, rel, resolve

MAX_TOOL_RESULT_CHARS = 20_000
# The same budget expressed in tokens, which is what the local model's context
# is actually measured in. For ASCII the two caps coincide; for Japanese the
# character cap alone let ~20k tokens of tool output into a 16k num_ctx, where
# Ollama silently truncated it and the model edited a file it never fully saw.
MAX_TOOL_RESULT_TOKENS = 5_000


def schemas() -> list[dict[str, Any]]:
    def tool(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }

    return [
        tool(
            "read_file",
            "Read a text file from the workspace. Use start_line/end_line for large files.",
            {
                "path": {"type": "string", "description": "Path relative to workspace root."},
                "start_line": {"type": "integer", "description": "1-indexed, optional."},
                "end_line": {"type": "integer", "description": "1-indexed inclusive, optional."},
            },
            ["path"],
        ),
        tool(
            "edit_file",
            "Replace an exact substring in a file. old_text must appear exactly once. "
            "This is the preferred way to change code.",
            {
                "path": {"type": "string"},
                "old_text": {"type": "string", "description": "Exact text to replace."},
                "new_text": {"type": "string", "description": "Replacement text."},
            },
            ["path", "old_text", "new_text"],
        ),
        tool(
            "write_file",
            "Write a file's full contents, creating it if needed. Use only for new files or "
            "full rewrites; prefer edit_file otherwise.",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
        tool(
            "grep",
            "Search the workspace with a regular expression. Returns matching lines with paths.",
            {
                "pattern": {"type": "string"},
                "glob": {"type": "string", "description": "Optional file filter, e.g. *.py"},
                "max_results": {"type": "integer", "description": "Default 80."},
            },
            ["pattern"],
        ),
        tool(
            "list_files",
            "List files under a directory (recursive, workspace-relative).",
            {"path": {"type": "string", "description": "Default is the workspace root."}},
            [],
        ),
    ]


def read_text(path: Path, errors: str = "surrogateescape") -> str:
    """Read preserving the file's own line endings.

    `Path.read_text` applies universal newlines, and `Path.write_text` then
    re-encodes them to os.linesep. On Windows that silently rewrites every LF
    file to CRLF, turning a one-line edit into a whole-file diff. Both sides
    pass newline="" so bytes round-trip unchanged.
    """
    with path.open("r", encoding="utf-8", errors=errors, newline="") as handle:
        return handle.read()


def write_text(path: Path, text: str, errors: str = "surrogateescape") -> None:
    with path.open("w", encoding="utf-8", errors=errors, newline="") as handle:
        handle.write(text)



@dataclass
class ToolBelt:
    cfg: Config
    read_only: bool = False
    originals: dict[Path, str | None] = field(default_factory=dict)
    touched: set[Path] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)

    # -- dispatch ---------------------------------------------------------
    def run(self, name: str, args: dict[str, Any]) -> str:
        self.calls.append(name)
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            return f"ERROR: unknown tool {name!r}"
        try:
            return _truncate(handler(**args))
        except SandboxError as exc:
            return f"ERROR: {exc}"
        except TypeError as exc:
            return f"ERROR: bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced back to the local model
            return f"ERROR: {type(exc).__name__}: {exc}"

    # -- snapshot / rollback ---------------------------------------------
    def _snapshot(self, path: Path) -> None:
        if path in self.originals:
            return
        self.originals[path] = (
            read_text(path) if path.is_file() else None
        )

    def rollback(self) -> list[str]:
        restored: list[str] = []
        for path, original in self.originals.items():
            if original is None:
                if path.is_file():
                    path.unlink()
                    restored.append(rel(self.cfg, path))
            else:
                write_text(path, original)
                restored.append(rel(self.cfg, path))
        self.touched.clear()
        return restored

    def diffstat(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted(self.touched):
            original = self.originals.get(path)
            new = read_text(path) if path.is_file() else ""
            added, removed = _line_delta((original or "").splitlines(), new.splitlines())
            out.append(
                {
                    "path": rel(self.cfg, path),
                    "status": "created" if original is None else "modified",
                    "added": added,
                    "removed": removed,
                }
            )
        return out

    # -- tools ------------------------------------------------------------
    def _t_read_file(
        self, path: str, start_line: int | None = None, end_line: int | None = None
    ) -> str:
        target = resolve(self.cfg, path, must_exist=True)
        if target.stat().st_size > self.cfg.limits.max_file_bytes:
            return (
                f"ERROR: {rel(self.cfg, target)} is larger than "
                f"{self.cfg.limits.max_file_bytes} bytes; read a line range instead."
            )
        lines = read_text(target, errors="replace").splitlines()
        lo = max(1, start_line or 1)
        hi = min(len(lines), end_line or len(lines))
        body = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(lo, hi + 1))
        return f"{rel(self.cfg, target)} (lines {lo}-{hi} of {len(lines)})\n{body}"

    def _t_edit_file(self, path: str, old_text: str, new_text: str) -> str:
        if self.read_only:
            return "ERROR: this task is read-only; no edits allowed."
        target = resolve(self.cfg, path, must_exist=True)
        content = read_text(target)
        count = content.count(old_text)
        if count == 0:
            return "ERROR: old_text not found. Re-read the file and copy the exact text."
        if count > 1:
            return f"ERROR: old_text appears {count} times; include more surrounding context."
        self._snapshot(target)
        write_text(target, content.replace(old_text, new_text, 1))
        self.touched.add(target)
        return f"OK: edited {rel(self.cfg, target)}"

    def _t_write_file(self, path: str, content: str) -> str:
        if self.read_only:
            return "ERROR: this task is read-only; no edits allowed."
        target = resolve(self.cfg, path)
        self._snapshot(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text(target, content)
        self.touched.add(target)
        # byte_length, not len(): a Japanese file is ~3x its character count in
        # UTF-8, and a receipt that says "bytes" should not mean "characters".
        return f"OK: wrote {rel(self.cfg, target)} ({encoding.byte_length(content)} bytes)"

    def _t_grep(self, pattern: str, glob: str | None = None, max_results: int = 80) -> str:
        limit = max(1, max_results)
        hits = _ripgrep(self.cfg, pattern, glob, limit)
        if hits is None:
            hits = self._python_grep(pattern, glob, limit)
        if isinstance(hits, str):
            return hits
        return "\n".join(hits) if hits else "no matches"

    def _python_grep(self, pattern: str, glob: str | None, limit: int) -> list[str] | str:
        """Fallback when ripgrep is absent -- notably plain Windows, where `rg`
        often exists only inside a bundled Git Bash and not on the system PATH."""
        # Smart-case, as ripgrep does it: case-insensitive unless the pattern
        # itself contains an uppercase letter. A local model searching for
        # "threshold" should find FREE_THRESHOLD.
        flags = 0 if any(c.isupper() for c in pattern) else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return f"ERROR: invalid pattern: {exc}"

        denied = {d.lower() for d in self.cfg.sandbox.deny}
        hits: list[str] = []
        for item in sorted(self.cfg.workspace.rglob(glob or "*")):
            if not item.is_file():
                continue
            parts = {p.lower() for p in item.relative_to(self.cfg.workspace).parts}
            if parts & denied or item.stat().st_size > self.cfg.limits.max_file_bytes:
                continue
            try:
                text = read_text(item, errors="strict")
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{rel(self.cfg, item)}:{number}:{line[:300]}")
                    if len(hits) >= limit:
                        return hits
        return hits

    def _t_list_files(self, path: str = ".") -> str:
        root = resolve(self.cfg, path, must_exist=True)
        denied = {d.lower() for d in self.cfg.sandbox.deny}
        out: list[str] = []
        for item in sorted(root.rglob("*")):
            if not item.is_file():
                continue
            parts = {p.lower() for p in item.relative_to(self.cfg.workspace).parts}
            if parts & denied:
                continue
            out.append(rel(self.cfg, item))
            if len(out) >= 400:
                out.append("... (truncated)")
                break
        return "\n".join(out) or "(empty)"


def _ripgrep(cfg: Config, pattern: str, glob: str | None, limit: int) -> list[str] | str | None:
    """Return hits, an error string, or None if ripgrep is unavailable."""
    if not shutil.which("rg"):
        return None
    cmd = ["rg", "--line-number", "--no-heading", "--color", "never", "--smart-case", "-m", "40"]
    if glob:
        cmd += ["--glob", glob]
    cmd += ["-e", pattern, "."]
    try:
        # Bytes, not text=True: ripgrep emits UTF-8 for a match in a Japanese
        # source file no matter what the console codepage is, and decoding that
        # with cp932 raises out of a function whose callers expect a value.
        proc = subprocess.run(
            cmd,
            cwd=cfg.workspace,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode not in (0, 1):
        return f"ERROR: search failed: {encoding.decode_output(proc.stderr)[:400]}"
    return encoding.decode_output(proc.stdout).splitlines()[:limit]


def _line_delta(old: list[str], new: list[str]) -> tuple[int, int]:
    added = removed = 0
    for line in difflib.unified_diff(old, new, n=0, lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def _truncate(text: str) -> str:
    """Cap a tool result on characters *and* on estimated tokens.

    The character cap bounds the transcript; the token cap bounds the local
    model's context window, and only the second one is meaningful for CJK text.
    Whichever bites first wins.
    """
    chars, tokens = len(text), encoding.estimate_tokens(text)
    if chars <= MAX_TOOL_RESULT_CHARS and tokens <= MAX_TOOL_RESULT_TOKENS:
        return text

    keep = min(chars, MAX_TOOL_RESULT_CHARS)
    if tokens > MAX_TOOL_RESULT_TOKENS:
        # Tokens track characters closely enough within a single writing system
        # for one proportional cut to land inside the budget.
        keep = min(keep, max(1, chars * MAX_TOOL_RESULT_TOKENS // tokens))
    return text[:keep] + (
        f"\n... (truncated at {MAX_TOOL_RESULT_CHARS} chars / "
        f"~{MAX_TOOL_RESULT_TOKENS} tokens)"
    )


def parse_args(raw: Any) -> dict[str, Any]:
    """Ollama returns arguments as a dict, but some templates emit a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
