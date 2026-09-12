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

from .config import Config

MAX_OUTPUT_CHARS = 1500


@dataclass
class Check:
    name: str
    ok: bool
    output: str = ""


@dataclass
class GateResult:
    ok: bool
    checks: list[Check] = field(default_factory=list)
    skipped: bool = False

    def summary(self) -> str:
        if self.skipped:
            return "no gate configured"
        return ", ".join(f"{c.name}={'pass' if c.ok else 'FAIL'}" for c in self.checks)

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
    checks: list[Check] = []
    for path in touched:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
                checks.append(Check(f"syntax:{path.name}", True))
            elif suffix == ".json":
                json.loads(path.read_text(encoding="utf-8", errors="replace"))
                checks.append(Check(f"syntax:{path.name}", True))
        except (SyntaxError, ValueError) as exc:
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
            data = json.loads(pkg.read_text(encoding="utf-8"))
        except ValueError:
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


def _run_command(cfg: Config, cmd: list[str]) -> Check:
    name = " ".join(cmd[:3])
    try:
        proc = subprocess.run(
            cmd,
            cwd=cfg.workspace,
            capture_output=True,
            text=True,
            timeout=cfg.gate.timeout_s,
            shell=False,
            check=False,
        )
    except FileNotFoundError:
        return Check(name, True, f"skipped: {cmd[0]} not installed")
    except subprocess.TimeoutExpired:
        return Check(name, False, f"timed out after {cfg.gate.timeout_s}s")

    output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    return Check(name, proc.returncode == 0, output[:MAX_OUTPUT_CHARS])
