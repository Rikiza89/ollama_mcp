"""MCP stdio server: delegate mechanical work to a local Ollama model.

Design constraints that shape everything here:

* **Four tools, not twelve.** Every tool schema is re-sent to the orchestrator on
  every single request, forever. A wide tool belt eats back the savings it exists
  to create.
* **Paths in, receipts out.** No tool takes file *content* as an argument and no
  tool returns file content. If the orchestrator has to read a file to write the
  delegation prompt, the tokens are already spent.
* **Nothing succeeds without the gate.** See `gate.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from typing import Literal

from mcp.server.mcpserver import MCPServer

from . import __version__, agent, config, i18n, metrics
from . import gate as gate_mod
from .ollama_client import OllamaClient, OllamaError

mcp = MCPServer("ollama-local", version=__version__)

Tier = Literal["fast", "deep"]


def _receipt(outcome: agent.Outcome, *, header: str) -> str:
    """Render the receipt the orchestrator sees.

    The status tokens -- APPLIED, NOT APPLIED, ESCALATE -- and the `gate:` and
    `local:` keys are protocol and stay ASCII in every language. Published
    `CLAUDE.md` delegation policies say "a tool returning ESCALATE means the
    working tree is unchanged", and translating that word would quietly break
    every one of them. Only the sentences around the tokens are localized.
    """
    strings = i18n.strings(outcome.language)
    lines = [header]
    if outcome.files:
        for entry in outcome.files:
            lines.append(
                f"  {entry['status']:>8}  {entry['path']}  +{entry['added']}/-{entry['removed']}"
            )
    if outcome.gate_summary:
        lines.append(f"gate: {outcome.gate_summary}")
    lines.append(
        f"local: {outcome.model}, {outcome.iterations} {strings.steps}, "
        f"{outcome.duration_ms / 1000:.1f}s, "
        f"{outcome.local_tokens_estimated} {strings.tokens_read_locally}"
    )
    if outcome.escalated:
        lines.append(f"ESCALATE: {outcome.reason}")
        if outcome.gate_failures:
            lines.append(strings.verifier_said)
            lines.append(outcome.gate_failures[:1200])
        lines.append(strings.tree_unchanged)
    return "\n".join(lines)


def _load(workspace_root: str) -> config.Config:
    return config.load(workspace_root)


@mcp.tool()
async def local_edit(
    workspace_root: str,
    instruction: str,
    tier: Tier = "fast",
) -> str:
    """Delegate a MECHANICAL code edit to the local model. USE THIS INSTEAD OF
    Read+Edit whenever the change is well-specified and does not need cross-file
    reasoning: renames, docstrings and comments, type annotations, adding a
    logging line, applying a pattern you already decided on, boilerplate, test
    scaffolding, formatting fixes.

    Do NOT read the files first -- that spends the tokens this tool exists to save.
    Pass file paths inside `instruction` and let the local model read them.

    The edit is applied to the working tree only if it passes the project's
    verification gate; otherwise it is rolled back and you get an ESCALATE.

    Args:
        workspace_root: Absolute path to the repository root.
        instruction: Self-contained task, naming the exact files and the exact
            change. The local model sees nothing else -- no conversation history.
        tier: "fast" for mechanical/high-volume work, "deep" for edits needing
            real code reasoning (slower, larger model).

    Returns:
        A receipt: files changed with line counts, gate verdict, timing. Never
        file content.
    """
    cfg = _load(workspace_root)
    outcome = await agent.run_task(
        cfg,
        tool_name="local_edit",
        instruction=instruction,
        tier=tier,
        read_only=False,
    )
    header = "APPLIED" if outcome.ok else "NOT APPLIED"
    if outcome.ok and outcome.answer:
        header = f"APPLIED: {outcome.answer}"
    return _receipt(outcome, header=header)


@mcp.tool()
async def local_explain(
    workspace_root: str,
    question: str,
    tier: Tier = "fast",
    answer_budget: int = 1200,
) -> str:
    """Ask the local model to read the repository and answer a FACTUAL question
    about it. USE THIS INSTEAD OF Read/Grep when you need to know what is in
    files but do not need the files themselves in context: "where is X defined",
    "what does module Y do", "which call sites pass argument Z", "summarize this
    log".

    Read-only: the local model has no write tools for this call.

    Args:
        workspace_root: Absolute path to the repository root.
        question: A specific, answerable question. Vague questions get ESCALATE.
        tier: "fast" (default) or "deep" for multi-file reasoning.
        answer_budget: Soft character cap on the answer. Keep it small.

    Returns:
        A dense answer with path:line citations, or ESCALATE. Never file dumps.
    """
    cfg = _load(workspace_root)
    outcome = await agent.run_task(
        cfg,
        tool_name="local_explain",
        instruction=question,
        tier=tier,
        read_only=True,
        answer_budget=answer_budget,
    )
    if outcome.escalated:
        return _receipt(outcome, header="NO ANSWER")
    body = outcome.answer[: answer_budget + 400]
    return f"{body}\n\n[local: {outcome.model}, {outcome.duration_ms / 1000:.1f}s]"


@mcp.tool()
async def local_verify(workspace_root: str, triage: bool = True) -> str:
    """Run this project's verification gate (from .ollama-mcp.toml, or autodetected)
    and return a COMPACT triage of any failures. USE THIS INSTEAD OF running
    lint/typecheck/test commands through Bash when you only need to know whether
    it passes and what broke -- raw tool output is often thousands of tokens.

    Args:
        workspace_root: Absolute path to the repository root.
        triage: If true and the gate fails, the local model summarizes the failures
            into a short actionable list instead of returning raw output.

    Returns:
        PASS, or a short list of what failed and where.
    """
    cfg = _load(workspace_root)
    # to_thread: the gate shells out for up to `timeout_s` (180s by default),
    # which would otherwise stall every other request on this stdio server.
    verdict = await asyncio.to_thread(gate_mod.run, cfg, [])
    strings = i18n.strings(i18n.resolve(cfg.i18n.language))
    if verdict.skipped:
        return strings.no_gate_configured.format(path=cfg.workspace / config.CONFIG_NAME)
    if verdict.ok:
        return f"PASS ({verdict.summary()})"

    raw = verdict.failures()
    if not triage:
        return f"FAIL ({verdict.summary()})\n{raw[:3000]}"

    outcome = await agent.run_task(
        cfg,
        tool_name="local_verify",
        instruction=strings.triage_instruction + raw[:12000],
        tier="fast",
        read_only=True,
        answer_budget=1500,
    )
    if outcome.escalated or not outcome.answer:
        return f"FAIL ({verdict.summary()})\n{raw[:3000]}"
    return f"FAIL ({verdict.summary()})\n\n{outcome.answer[:2000]}"


@mcp.tool()
async def local_status(workspace_root: str) -> str:
    """Check that local delegation is actually available and see what it has saved.
    Cheap -- no model inference. Call this once at the start of a session if you
    intend to delegate, and whenever a local tool fails unexpectedly.

    Args:
        workspace_root: Absolute path to the repository root.

    Returns:
        Ollama health, configured model tiers, whether they are installed, the
        gate configuration, and estimated tokens avoided so far.
    """
    cfg = _load(workspace_root)
    language = i18n.resolve(cfg.i18n.language)
    strings = i18n.strings(language)
    client = OllamaClient(cfg.ollama_host)
    try:
        health = await client.health()
    except OllamaError as exc:
        return f"UNAVAILABLE: {exc}\n{strings.ollama_unavailable}"

    installed = set(health["models"])
    tiers = []
    for label, name in (("fast", cfg.models.fast), ("deep", cfg.models.deep)):
        mark = strings.model_ok if name in installed else strings.model_missing.format(model=name)
        tiers.append(f"  {label}: {name} -- {mark}")

    commands = cfg.gate.commands or gate_mod.autodetect(cfg)
    gate_desc = (
        "\n".join(f"  {' '.join(c)}" for c in commands) if commands else f"  {strings.gate_syntax_only}"
    )

    # Surfaced so a delegation answering in an unexpected language is one call to
    # diagnose, rather than a mystery about the model.
    configured = cfg.i18n.language
    resolved = f"{language.value} (from {configured})" if configured == i18n.AUTO else language.value

    return "\n".join(
        [
            f"Ollama {health['version']} at {cfg.ollama_host}",
            f"{strings.label_config}: {cfg.source}",
            f"{strings.label_language}: {resolved}",
            f"{strings.label_models}:",
            *tiers,
            f"{strings.label_gate}:",
            gate_desc,
            f"{strings.label_savings}: " + json.dumps(metrics.summarize(cfg), ensure_ascii=False),
        ]
    )


def main() -> None:
    """Entry point for the `ollama-mcp` console script (stdio transport)."""
    # The MCP transport already forces UTF-8 on stdin/stdout, but stderr keeps the
    # platform encoding. On a non-UTF-8 console -- cp932, cp1252 -- a traceback or
    # log line containing a non-ASCII path or a delegated Japanese instruction
    # would raise UnicodeEncodeError from inside the error handler and take the
    # server down. Degrade to replacement characters instead.
    if sys.stderr is not None and (sys.stderr.encoding or "").lower() not in {
        "utf-8",
        "utf8",
    }:
        with contextlib.suppress(AttributeError, ValueError):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
