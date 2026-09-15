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
from .sandbox import SandboxError, denied_rule, rel, resolve

MAX_TOOL_RESULT_CHARS = 20_000
# The same budget expressed in tokens, which is what the local model's context
# is actually measured in. For ASCII the two caps coincide; for Japanese the
# character cap alone let ~20k tokens of tool output into a 16k num_ctx, where
# Ollama silently truncated it and the model edited a file it never fully saw.
MAX_TOOL_RESULT_TOKENS = 5_000

# The verdict tool. Handled in the agent loop rather than by the belt: calling it
# ends the task, so it has no result to hand back to the model. Named here so the
# loop and the schema cannot drift apart.
FINISH = "finish"


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
            "Replace an exact substring in a file. Prefer a SHORT fragment from a "
            "single line, copied exactly. If that fragment appears more than once, "
            "pass near_line to say which one you mean. This is the preferred way "
            "to change code.",
            {
                "path": {"type": "string"},
                "old_text": {
                    "type": "string",
                    "description": "Exact text to replace. Keep it short and on one "
                    "line; the line numbers read_file shows are not part of the file.",
                },
                "new_text": {"type": "string", "description": "Replacement text."},
                "near_line": {
                    "type": "integer",
                    "description": "Optional. When old_text occurs several times, the "
                    "line number of the one you mean, as shown by read_file.",
                },
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
        tool(
            FINISH,
            "End the task. Call this exactly once, as your very last action. Use "
            'status="done" ONLY if you actually finished what was asked. Use '
            'status="escalate" for everything else -- the task is ambiguous, you '
            "could not do it, or you are not certain you did it right. Escalating "
            "is a correct outcome, not a failure; guessing is not.",
            {
                "status": {
                    "type": "string",
                    "enum": ["done", "escalate"],
                    "description": 'Exactly "done" or "escalate".',
                },
                "summary": {
                    "type": "string",
                    "description": "One sentence: what you changed, or what is blocking you.",
                },
            },
            ["status", "summary"],
        ),
    ]


def _parameter_types() -> dict[str, dict[str, str]]:
    return {
        tool["function"]["name"]: {
            key: spec.get("type", "string")
            for key, spec in tool["function"]["parameters"]["properties"].items()
        }
        for tool in schemas()
    }


PARAMETER_TYPES = _parameter_types()


def coerce_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Give text-channel arguments the types the schema declares.

    The native tool-calling channel delivers a typed JSON object. A call
    recovered from the message body does not: every value arrives as a string,
    so `start_line` turns up as "45" and `max(1, "45")` raises TypeError from
    inside the handler -- reported to the model as a bad-arguments error it has
    no way to act on, since the call it made was perfectly correct.
    """
    types = PARAMETER_TYPES.get(name, {})
    out: dict[str, Any] = {}
    for key, value in args.items():
        declared = types.get(key)
        if isinstance(value, str) and declared in {"integer", "number"}:
            try:
                value = int(value.strip()) if declared == "integer" else float(value.strip())
            except ValueError:
                pass  # leave it; the handler reports a better error than we can
        out[key] = value
    return out


def read_source(path: Path) -> tuple[str, str]:
    """Read a file as text plus the codec that decoded it.

    Binary I/O throughout, so nothing is translated on the way in or out: no
    universal-newline rewrite (which turned every LF file on Windows into a
    whole-file CRLF diff) and no encoding upgrade (`write_text` used to emit
    UTF-8 unconditionally, quietly converting a cp932 source file the task had
    only asked to add a comment to).
    """
    return encoding.detect_text(path.read_bytes())


def dominant_newline(text: str) -> str:
    """The line ending a file actually uses, measured from its own content."""
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def to_newline(text: str, style: str) -> str:
    """Re-punctuate `text` with one line ending, whatever it arrived with."""
    body = text.replace("\r\n", "\n")
    return body.replace("\n", style) if style != "\n" else body


def write_source(path: Path, text: str, codec: str) -> None:
    """Write `text` back in the codec the file was read with.

    Raises:
        encoding.UnrepresentableText: if the new text does not fit that codec.
    """
    path.write_bytes(encoding.encode_text(text, codec))



@dataclass
class ToolBelt:
    cfg: Config
    read_only: bool = False
    # Raw bytes, not decoded text: a rollback has to restore the file exactly as
    # it was, including its encoding, its BOM and its line endings, and the only
    # representation that guarantees that is the one that came off the disk.
    originals: dict[Path, bytes | None] = field(default_factory=dict)
    touched: set[Path] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)
    # Directories `write_file` had to create, deepest last, so a rollback can
    # take them back out instead of leaving empty scaffolding behind.
    created_dirs: list[Path] = field(default_factory=list)

    # -- dispatch ---------------------------------------------------------
    def run(self, name: str, args: dict[str, Any]) -> str:
        self.calls.append(name)
        args = coerce_args(name, args)
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            return f"ERROR: unknown tool {name!r}"
        try:
            return _truncate(handler(**args))
        except SandboxError as exc:
            return f"ERROR: {exc}"
        except encoding.UnrepresentableText as exc:
            return f"ERROR: {exc}"
        except TypeError as exc:
            return f"ERROR: bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced back to the local model
            return f"ERROR: {type(exc).__name__}: {exc}"

    # -- snapshot / rollback ---------------------------------------------
    def _snapshot(self, path: Path) -> None:
        if path in self.originals:
            return
        self.originals[path] = path.read_bytes() if path.is_file() else None

    def rollback(self) -> list[str]:
        restored: list[str] = []
        for path, original in self.originals.items():
            if original is None:
                if path.is_file():
                    path.unlink()
                    restored.append(rel(self.cfg, path))
            else:
                path.write_bytes(original)
                restored.append(rel(self.cfg, path))
        self._remove_created_dirs()
        self.touched.clear()
        return restored

    def _make_parents(self, target: Path) -> None:
        """Create the parents `write_file` needs, remembering the new ones."""
        missing: list[Path] = []
        parent = target.parent
        while not parent.exists() and self.cfg.workspace in parent.parents:
            missing.append(parent)
            parent = parent.parent
        target.parent.mkdir(parents=True, exist_ok=True)
        self.created_dirs.extend(reversed(missing))

    def _remove_created_dirs(self) -> None:
        """Take back directories `write_file` created, deepest first.

        Only ever removes a directory this belt made and that is now empty, so
        an escalation leaves no trace -- "the working tree is unchanged" should
        not come with a litter of empty folders.
        """
        for directory in sorted(self.created_dirs, reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass  # not empty, or already gone: either way, leave it
        self.created_dirs.clear()

    def diffstat(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted(self.touched):
            original = self.originals.get(path)
            before = encoding.detect_text(original)[0] if original else ""
            after = read_source(path)[0] if path.is_file() else ""
            added, removed = _line_delta(before.splitlines(), after.splitlines())
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
        raw = target.read_bytes()
        if encoding.looks_binary(raw):
            return f"ERROR: {rel(self.cfg, target)} looks like a binary file."
        lines = encoding.detect_text(raw)[0].splitlines()
        total = len(lines)

        # An out-of-range window used to produce a header reading "lines 99-2 of
        # 2" and an empty body -- no error, nothing to act on, and a small model
        # reliably just asks for it again. Say what went wrong instead.
        lo = max(1, start_line or 1)
        hi = min(total, end_line or total)
        if lo > total:
            return f"ERROR: {rel(self.cfg, target)} has only {total} lines; start_line={lo}."
        if lo > hi:
            return f"ERROR: start_line={lo} is after end_line={end_line}."

        body = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(lo, hi + 1))
        return f"{rel(self.cfg, target)} (lines {lo}-{hi} of {total})\n{body}"

    def _t_edit_file(
        self, path: str, old_text: str, new_text: str, near_line: int | None = None
    ) -> str:
        if self.read_only:
            return "ERROR: this task is read-only; no edits allowed."
        target = resolve(self.cfg, path, must_exist=True)
        content, codec = read_source(target)
        count = content.count(old_text)
        if count == 0:
            # The message is the model's only feedback channel, so it names the
            # actual cause. Watching a 30B work, every failed edit was a
            # multi-line block reassembled from read_file's numbered output --
            # and it would then retry the very same string two or three times.
            return (
                "ERROR: old_text not found. The line numbers read_file shows are NOT "
                "part of the file, so a whole line copied from that output will not "
                "match. Use a SHORT fragment from a single line instead -- for "
                "Japanese text, just the Japanese characters themselves, with no "
                "quotes, indentation or line numbers. Do not retry the same old_text."
            )
        if count > 1:
            if near_line is None:
                return (
                    f"ERROR: old_text appears {count} times. Do not add surrounding "
                    f"lines -- call edit_file again with the same old_text plus "
                    f"near_line=<the line number read_file showed for the one you mean>."
                )
            offset = _nth_near(content, old_text, near_line)
            if offset is None:
                return f"ERROR: no occurrence of old_text near line {near_line}."
            self._snapshot(target)
            new_text = to_newline(new_text, dominant_newline(content))
            edited = content[:offset] + new_text + content[offset + len(old_text) :]
            write_source(target, edited, codec)
            self.touched.add(target)
            return f"OK: edited {rel(self.cfg, target)} near line {near_line}"
        self._snapshot(target)
        # Same reason: a model writing a multi-line replacement types LF even
        # when every other line in the file ends CRLF.
        new_text = to_newline(new_text, dominant_newline(content))
        write_source(target, content.replace(old_text, new_text, 1), codec)
        self.touched.add(target)
        return f"OK: edited {rel(self.cfg, target)}"

    def _t_write_file(self, path: str, content: str) -> str:
        if self.read_only:
            return "ERROR: this task is read-only; no edits allowed."
        target = resolve(self.cfg, path)
        codec = "utf-8"
        if target.is_file():
            existing, codec = read_source(target)
            # A model emits LF. Writing that straight back flips a CRLF file
            # wholesale, which is a whole-file diff for a one-line change and
            # breaks any path pinned by .gitattributes (`*.html -text`,
            # `*.bat eol=crlf`). edit_file has always preserved endings; a full
            # rewrite has to as well, or the documented guarantee is only half
            # true. A file being created keeps whatever the model wrote.
            content = to_newline(content, dominant_newline(existing))
        self._snapshot(target)
        self._make_parents(target)
        write_source(target, content, codec)
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

        hits: list[str] = []
        for item in sorted(self.cfg.workspace.rglob(glob or "*")):
            if not item.is_file() or denied_rule(self.cfg, item) is not None:
                continue
            if item.stat().st_size > self.cfg.limits.max_file_bytes:
                continue
            try:
                raw = item.read_bytes()
            except OSError:
                continue
            if encoding.looks_binary(raw):
                continue
            # detect_text, not strict UTF-8: the old test quietly excluded every
            # cp932 source file in the repository from search, as though those
            # files simply contained no matches.
            text = encoding.detect_text(raw)[0]
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{rel(self.cfg, item)}:{number}:{line[:300]}")
                    if len(hits) >= limit:
                        return hits
        return hits

    def _t_list_files(self, path: str = ".") -> str:
        root = resolve(self.cfg, path, must_exist=True)
        out: list[str] = []
        for item in sorted(root.rglob("*")):
            if not item.is_file() or denied_rule(self.cfg, item) is not None:
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
    for token in cfg.sandbox.deny:
        cmd += ["--glob", f"!**/{token}", "--glob", f"!**/{token}/**"]
        if token.startswith("."):
            cmd += ["--glob", f"!**/*{token}"]
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
    return _keep_allowed(cfg, encoding.decode_output(proc.stdout).splitlines())[:limit]


def _keep_allowed(cfg: Config, lines: list[str]) -> list[str]:
    """Drop hits whose path hits a deny rule.

    The --glob exclusions handed to ripgrep should already have kept it away
    from these files, but the deny list is a guard, and a guard does not stake
    itself on another program's glob dialect agreeing with ours -- this path
    used to apply no deny filtering whatever, so with `rg` installed the whole
    list was unenforced for search. A hit whose path will not parse is dropped
    as well: failing closed costs at most one missed match, failing open costs
    a private key.
    """
    out: list[str] = []
    for line in lines:
        path, separator, _ = line.partition(":")
        if not separator:
            continue
        try:
            resolved = (cfg.workspace / path).resolve()
        except OSError:  # pragma: no cover - platform dependent
            continue
        if resolved != cfg.workspace and cfg.workspace not in resolved.parents:
            continue
        if denied_rule(cfg, resolved) is None:
            out.append(line)
    return out


def _nth_near(content: str, needle: str, near_line: int) -> int | None:
    """Offset of the occurrence of `needle` whose line is closest to `near_line`."""
    best: tuple[int, int] | None = None
    start = 0
    while (found := content.find(needle, start)) != -1:
        line = content.count(chr(10), 0, found) + 1
        distance = abs(line - near_line)
        if best is None or distance < best[0]:
            best = (distance, found)
        start = found + 1
    return None if best is None else best[1]


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
