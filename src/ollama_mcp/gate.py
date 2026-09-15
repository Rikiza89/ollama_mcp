"""The verification gate.

This is what makes "same result as pure Claude Code" achievable rather than
aspirational. Parity does not come from the local model being good; it comes
from a narrow task plus a machine-checkable result. Nothing is reported as a
success until it passes here.

Two layers:
  1. Syntax check of every file the local model touched (always, free, universal).
  2. The project's own check commands, from `.ollama-mcp.toml` -- or autodetected
     if the project has no config.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import config, encoding
from .config import Config

MAX_OUTPUT_CHARS = 1500


@dataclass
class Check:
    name: str
    ok: bool
    output: str = ""
    # False when the command could not be executed at all. Kept apart from a
    # plain failure so the receipt can say "your gate is broken" rather than
    # "your code is broken" -- but it is still not a pass. See `_run_command`.
    ran: bool = True

    @property
    def verdict(self) -> str:
        if self.ok:
            return "pass"
        return "FAIL" if self.ran else "UNVERIFIED"


@dataclass
class GateResult:
    ok: bool
    checks: list[Check] = field(default_factory=list)
    skipped: bool = False

    def summary(self) -> str:
        if self.skipped:
            return "no gate configured"
        return ", ".join(f"{c.name}={c.verdict}" for c in self.checks)

    def failures(self) -> str:
        return "\n\n".join(f"[{c.name}]\n{c.output}".strip() for c in self.checks if not c.ok)


def run(cfg: Config, touched: list[Path]) -> GateResult:
    checks: list[Check] = []

    syntax = _syntax_checks(touched)
    checks.extend(syntax)

    commands = cfg.gate.commands or (autodetect(cfg) if cfg.gate.autodetect else [])
    for cmd in commands:
        checks.append(_run_command(cfg, cmd))

    return GateResult(
        ok=all(c.ok for c in checks),
        checks=checks,
        skipped=not checks,
    )


def _syntax_checks(touched: list[Path]) -> list[Check]:
    """Parse every touched file we know how to parse.

    Both parsers are fed *bytes*, never pre-decoded text. That is what makes the
    check survive a file a Japanese editor wrote: CPython's tokenizer strips a
    UTF-8 BOM and honours a PEP 263 coding cookie (`# -*- coding: cp932 -*-`),
    while the same content decoded as UTF-8 str arrives with a leading U+FEFF
    and fails to parse -- a false failure, which costs a correct edit a rollback
    and a needless escalation.
    """
    checks: list[Check] = []
    for path in touched:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".py", ".json"}:
            continue
        try:
            raw = path.read_bytes()
            if suffix == ".py":
                ast.parse(raw, filename=str(path))
            else:
                # json.loads autodetects UTF-8/16/32 from bytes per RFC 8259 but
                # still rejects a UTF-8 BOM, so strip that one ourselves.
                json.loads(encoding.strip_bom(raw))
            checks.append(Check(f"syntax:{path.name}", True))
        except (SyntaxError, ValueError, OSError) as exc:
            checks.append(Check(f"syntax:{path.name}", False, str(exc)[:MAX_OUTPUT_CHARS]))
    return checks


def autodetect(cfg: Config) -> list[list[str]]:
    """Best-effort project check. Conservative: only commands that are cheap and
    that a repo is very likely to tolerate."""
    root = cfg.workspace
    commands: list[list[str]] = []

    is_python = (root / "pyproject.toml").is_file() or any(root.glob("*.py"))
    if is_python and shutil.which("ruff"):
        commands.append(["ruff", "check", "--quiet", "."])

    pkg = root / "package.json"
    if pkg.is_file():
        try:
            data = json.loads(encoding.strip_bom(pkg.read_bytes()))
        except (ValueError, OSError):
            data = {}
        scripts = data.get("scripts", {}) or {}
        if "typecheck" in scripts:
            commands.append(["npm", "run", "--silent", "typecheck"])
        elif (root / "tsconfig.json").is_file() and shutil.which("npx"):
            commands.append(["npx", "--no-install", "tsc", "--noEmit"])

    if (root / "Cargo.toml").is_file() and shutil.which("cargo"):
        commands.append(["cargo", "check", "--quiet"])

    if (root / "go.mod").is_file() and shutil.which("go"):
        commands.append(["go", "build", "./..."])

    return commands


def _resolve_executable(cfg: Config, program: str) -> str:
    """Resolve a workspace-relative interpreter or script path.

    On Windows `CreateProcess` looks the executable up against the *parent
    process's* directory, not the `cwd=` it is handed (POSIX gets this right:
    CPython chdirs in the child before exec). So a natural gate command like
    `["./.venv/bin/python", "-m", "pytest"]` or `["python-embed/python.exe", ...]`
    would fail with FileNotFoundError even though the file is right there in the
    project. Bare names (`ruff`, `npm`) are left alone for PATH lookup.
    """
    if "/" not in program and "\\" not in program:
        return program
    candidate = Path(program)
    if candidate.is_absolute():
        return program
    resolved = (cfg.workspace / candidate).resolve()
    return str(resolved) if resolved.is_file() else program


def _run_command(cfg: Config, cmd: list[str]) -> Check:
    name = " ".join(cmd[:3])
    cmd = [_resolve_executable(cfg, cmd[0]), *cmd[1:]]
    try:
        # Bytes, not text=True. `text=True` decodes with the locale encoding,
        # which is cp932 on a Japanese Windows install; ruff, pytest and tsc all
        # emit UTF-8 whatever the console codepage is, so the first non-ASCII
        # byte raised UnicodeDecodeError from here -- uncaught, killing the whole
        # delegation rather than merely failing the check.
        proc = subprocess.run(
            cmd,
            cwd=cfg.workspace,
            capture_output=True,
            timeout=cfg.gate.timeout_s,
            shell=False,
            check=False,
        )
    except FileNotFoundError:
        # NOT a pass. This used to return ok=True, so a typo in `commands`, or an
        # MCP server launched with a PATH that lacks the tool -- the normal case,
        # since the server is spawned from the desktop app's environment and not
        # from your shell -- turned the gate into a no-op that still reported
        # `gate: pass`. The gate is the whole basis for trusting a delegated
        # edit; a check that did not run has not passed.
        return Check(
            name,
            False,
            f"{cmd[0]} could not be run (not installed, or not on this server's "
            f"PATH), so this check verified nothing. Fix the command in "
            f"{config.CONFIG_NAME}, or remove it.",
            ran=False,
        )
    except subprocess.TimeoutExpired:
        return Check(name, False, f"timed out after {cfg.gate.timeout_s}s")
    except OSError as exc:
        return Check(name, False, f"could not run {cmd[0]}: {exc}")

    parts = (encoding.decode_output(proc.stdout), encoding.decode_output(proc.stderr))
    output = "\n".join(part for part in parts if part).strip()
    return Check(name, proc.returncode == 0, output[:MAX_OUTPUT_CHARS])
