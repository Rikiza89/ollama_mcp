"""The local agentic loop.

Claude delegates a task; this runs it against Ollama with a small local tool
belt, verifies the result, and returns a *receipt*. The bulk text -- file
bodies, search hits, tool results -- lives and dies inside this function. That
containment is the entire point: it is what converts "a second model" into
"token savings".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import gate as gate_mod
from . import localtools, metrics, toolcalls
from .config import Config
from .ollama_client import ChatResult, OllamaClient, OllamaError

EDIT_SYSTEM = """You are a local code-editing assistant working inside a real repository.

Rules, in order of importance:
1. Do exactly what the task says. Do not refactor, rename, reformat, or "improve"
   anything you were not asked to change.
2. Always read a file before editing it. Copy `old_text` for edit_file verbatim from
   what you read, including indentation.
3. Prefer edit_file over write_file. Use write_file only for new files.
4. Keep the surrounding style: same naming, same comment density, same idioms.
5. When you are done, reply with plain text only -- no tool call -- in this shape:
   DONE: <one sentence on what you changed>
   If the task is ambiguous, impossible, or needs judgement you are not sure about,
   reply instead with:
   ESCALATE: <one sentence on exactly what is blocking you>
   Escalating is a correct outcome, not a failure. Never guess."""

READ_SYSTEM = """You are a local code-reading assistant working inside a real repository.

Rules:
1. Answer only from what you actually read with the tools. Never invent file names,
   symbols, or line numbers.
2. Read narrowly: grep first, then read only the relevant line ranges.
3. You must not modify anything. You have no write tools.
4. When done, reply with plain text only -- no tool call -- starting with:
   DONE: <your answer>
   Be dense and specific. Cite paths as path:line. Keep it under {budget} characters.
   If you cannot answer from the repository, reply:
   ESCALATE: <what is missing>"""


@dataclass
class Outcome:
    ok: bool
    escalated: bool
    reason: str = ""
    answer: str = ""
    files: list[dict[str, Any]] = field(default_factory=list)
    gate_summary: str = ""
    gate_failures: str = ""
    iterations: int = 0
    model: str = ""
    duration_ms: int = 0
    tool_calls: list[str] = field(default_factory=list)
    local_prompt_tokens: int = 0
    local_completion_tokens: int = 0
    local_chars_consumed: int = 0
    recovered_calls: int = 0


def pick_model(cfg: Config, tier: str) -> tuple[str, int]:
    if tier == "deep":
        return cfg.models.deep, cfg.models.deep_num_ctx
    return cfg.models.fast, cfg.models.fast_num_ctx


async def run_task(
    cfg: Config,
    *,
    tool_name: str,
    instruction: str,
    tier: str,
    read_only: bool,
    answer_budget: int = 1200,
) -> Outcome:
    """Run one delegated task end to end, including the gate and one retry."""
    model, num_ctx = pick_model(cfg, tier)
    client = OllamaClient(cfg.ollama_host, timeout_s=cfg.limits.request_timeout_s)
    belt = localtools.ToolBelt(cfg=cfg, read_only=read_only)

    system = (
        READ_SYSTEM.format(budget=answer_budget)
        if read_only
        else EDIT_SYSTEM
    )
    tools = [
        schema
        for schema in localtools.schemas()
        if not (read_only and schema["function"]["name"] in {"edit_file", "write_file"})
    ]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Workspace root: {cfg.workspace}\n\nTask:\n{instruction}"},
    ]

    started = time.monotonic()
    outcome = Outcome(ok=False, escalated=False, model=model)

    try:
        outcome = await _loop(client, cfg, belt, messages, model, num_ctx, tools, outcome)
    except OllamaError as exc:
        outcome.escalated = True
        outcome.reason = str(exc)

    if not read_only and belt.touched and not outcome.escalated:
        verdict = gate_mod.run(cfg, sorted(belt.touched))
        outcome.gate_summary = verdict.summary()
        if not verdict.ok:
            outcome.gate_failures = verdict.failures()
            retried = False
            if cfg.limits.max_local_retries > 0:
                retried = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your change failed verification. Fix it, or reply "
                            "ESCALATE if you cannot.\n\n" + verdict.failures()
                        ),
                    }
                )
                outcome = await _loop(
                    client, cfg, belt, messages, model, num_ctx, tools, outcome
                )
                verdict = gate_mod.run(cfg, sorted(belt.touched))
                outcome.gate_summary = verdict.summary()
                outcome.gate_failures = verdict.failures()

            if not verdict.ok:
                restored = belt.rollback()
                outcome.ok = False
                outcome.escalated = True
                outcome.reason = (
                    f"verification failed{' after one local retry' if retried else ''}; "
                    f"rolled back {len(restored)} file(s)"
                )
        if verdict.ok and not outcome.escalated:
            outcome.ok = True

    if read_only and not outcome.escalated:
        outcome.ok = bool(outcome.answer)

    if not read_only and not belt.touched and not outcome.escalated:
        # Small models sometimes narrate the change and report DONE without ever
        # calling edit_file. One pointed nudge recovers most of those; escalating
        # straight away would waste the orchestrator's time on work that is
        # actually within local reach.
        messages.append(
            {
                "role": "user",
                "content": (
                    "You reported DONE but the working tree is unchanged -- you never "
                    "called edit_file or write_file. Describing the change is not making "
                    "it. Make the edit now with the tools, or reply ESCALATE with the "
                    "reason you cannot."
                ),
            }
        )
        outcome = await _loop(client, cfg, belt, messages, model, num_ctx, tools, outcome)
        if belt.touched and not outcome.escalated:
            verdict = gate_mod.run(cfg, sorted(belt.touched))
            outcome.gate_summary = verdict.summary()
            if verdict.ok:
                outcome.ok = True
            else:
                outcome.gate_failures = verdict.failures()
                restored = belt.rollback()
                outcome.escalated = True
                outcome.reason = f"verification failed; rolled back {len(restored)} file(s)"

    if not read_only and not belt.touched and not outcome.escalated:
        outcome.ok = False
        outcome.escalated = True
        outcome.reason = outcome.reason or "local model made no changes"

    outcome.files = belt.diffstat() if not read_only else []
    outcome.tool_calls = belt.calls
    outcome.duration_ms = int((time.monotonic() - started) * 1000)

    metrics.append(
        cfg,
        metrics.Record(
            tool=tool_name,
            model=model,
            ok=outcome.ok,
            escalated=outcome.escalated,
            duration_ms=outcome.duration_ms,
            iterations=outcome.iterations,
            local_prompt_tokens=outcome.local_prompt_tokens,
            local_completion_tokens=outcome.local_completion_tokens,
            local_chars_consumed=outcome.local_chars_consumed,
            receipt_chars=len(outcome.answer) + len(outcome.reason) + 200,
            gate=outcome.gate_summary,
        ),
    )
    return outcome


async def _loop(
    client: OllamaClient,
    cfg: Config,
    belt: localtools.ToolBelt,
    messages: list[dict[str, Any]],
    model: str,
    num_ctx: int,
    tools: list[dict[str, Any]],
    outcome: Outcome,
) -> Outcome:
    for _ in range(cfg.limits.max_iterations):
        outcome.iterations += 1
        result: ChatResult = await client.chat(
            model=model,
            messages=messages,
            tools=tools,
            num_ctx=num_ctx,
            temperature=cfg.models.temperature,
            keep_alive=cfg.models.keep_alive,
        )
        outcome.local_prompt_tokens += result.prompt_tokens
        outcome.local_completion_tokens += result.completion_tokens

        calls = result.tool_calls
        if not calls:
            # Some models print the call instead of using the native channel.
            calls = toolcalls.extract(result.content, {t["function"]["name"] for t in tools})
            if calls:
                outcome.recovered_calls += len(calls)

        if not calls:
            text = result.content
            if text.upper().startswith("ESCALATE"):
                outcome.escalated = True
                outcome.reason = text.split(":", 1)[-1].strip() or text
            else:
                outcome.answer = text.split(":", 1)[-1].strip() if text.upper().startswith(
                    "DONE"
                ) else text
            return outcome

        messages.append({"role": "assistant", "content": result.content, "tool_calls": calls})
        for call in calls:
            fn = call.get("function", {}) or {}
            name = fn.get("name", "")
            args = localtools.parse_args(fn.get("arguments"))
            tool_result = belt.run(name, args)
            outcome.local_chars_consumed += len(tool_result)
            messages.append({"role": "tool", "name": name, "content": tool_result})

    outcome.escalated = True
    outcome.reason = f"local model hit the {cfg.limits.max_iterations}-step limit without finishing"
    return outcome
