"""The local agentic loop.

Claude delegates a task; this runs it against Ollama with a small local tool
belt, verifies the result, and returns a *receipt*. The bulk text -- file
bodies, search hits, tool results -- lives and dies inside this function. That
containment is the entire point: it is what converts "a second model" into
"token savings".
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from . import encoding, i18n, localtools, metrics, sentinels, toolcalls
from . import gate as gate_mod
from .config import Config
from .i18n import Strings
from .ollama_client import ChatResult, OllamaClient, OllamaError
from .sentinels import Verdict


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
    local_tokens_estimated: int = 0
    recovered_calls: int = 0
    # How many old tool results had to be dropped to keep the transcript inside
    # the local model's context window. Non-zero means it was working from a
    # partial view, which is worth knowing when a result looks wrong.
    elided_results: int = 0
    language: i18n.Language = i18n.Language.EN
    # True only when the local model gave an explicit success verdict. Never
    # inferred from the absence of an escalation -- see sentinels.py.
    finished: bool = False


# Fraction of num_ctx the transcript may occupy. The rest is headroom for the
# model's own reply and for whatever the chat template adds around each message.
CONTEXT_BUDGET = 0.75

ELIDED = "[earlier tool result dropped to fit the local model's context window]"


def _fit_context(messages: list[dict[str, Any]], tools: list[dict[str, Any]], num_ctx: int) -> int:
    """Blank out the oldest tool results until the transcript fits `num_ctx`.

    Capping each individual tool result, which `localtools` already does, does
    not cap their sum: twelve iterations of a 5,000-token result is 60,000
    tokens going into a 16,384-token window, where Ollama truncates it silently
    and the model edits a file it never fully saw. That silent truncation is the
    single failure this server most needs not to have, so the transcript gets a
    budget of its own.

    The messages themselves are kept and only their *content* is replaced, so
    every `tool` message still lines up with the `tool_calls` that produced it.
    Dropping them outright would reclaim a few more tokens and leave the chat
    template rendering an assistant turn whose calls have no results.

    Returns:
        The number of results newly elided.
    """
    budget = int(num_ctx * CONTEXT_BUDGET) - encoding.estimate_tokens(json.dumps(tools))
    total = sum(encoding.estimate_tokens(str(m.get("content") or "")) for m in messages)
    elided = 0
    for message in messages:  # oldest first
        if total <= budget:
            break
        if message.get("role") != "tool":
            continue
        content = str(message.get("content") or "")
        if content == ELIDED:
            continue
        total -= encoding.estimate_tokens(content) - encoding.estimate_tokens(ELIDED)
        message["content"] = ELIDED
        elided += 1
    return elided


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

    # Under the default `auto`, the instruction itself picks the language: a task
    # written in Japanese gets the Japanese prompt even on an English-locale
    # machine, which is the common case for a Japanese developer's laptop.
    language = i18n.resolve(cfg.i18n.language, hint=instruction)
    strings = i18n.strings(language)
    system = strings.read_system.format(budget=answer_budget) if read_only else strings.edit_system
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
    outcome = Outcome(ok=False, escalated=False, model=model, language=language)

    try:
        outcome = await _loop(client, cfg, belt, messages, model, num_ctx, tools, outcome, strings)
    except OllamaError as exc:
        outcome.escalated = True
        outcome.reason = str(exc)

    if not read_only and belt.touched and not outcome.escalated:
        verdict = await asyncio.to_thread(gate_mod.run, cfg, sorted(belt.touched))
        outcome.gate_summary = verdict.summary()
        if not verdict.ok:
            outcome.gate_failures = verdict.failures()
            retried = False
            if cfg.limits.max_local_retries > 0:
                retried = True
                messages.append(
                    {
                        "role": "user",
                        "content": strings.gate_retry + verdict.failures(),
                    }
                )
                outcome = await _loop(
                    client, cfg, belt, messages, model, num_ctx, tools, outcome, strings
                )
                verdict = await asyncio.to_thread(gate_mod.run, cfg, sorted(belt.touched))
                outcome.gate_summary = verdict.summary()
                outcome.gate_failures = verdict.failures()

            if not verdict.ok:
                restored = belt.rollback()
                outcome.ok = False
                outcome.escalated = True
                outcome.reason = strings.verify_failed.format(
                    retry=strings.after_retry if retried else "",
                    count=len(restored),
                )
        if verdict.ok and outcome.finished and not outcome.escalated:
            outcome.ok = True

    if read_only and not outcome.escalated:
        outcome.ok = outcome.finished and bool(outcome.answer)

    if not read_only and not belt.touched and not outcome.escalated:
        # Small models sometimes narrate the change and report DONE without ever
        # calling edit_file. One pointed nudge recovers most of those; escalating
        # straight away would waste the orchestrator's time on work that is
        # actually within local reach.
        messages.append(
            {
                "role": "user",
                "content": strings.no_edit_nudge,
            }
        )
        outcome = await _loop(client, cfg, belt, messages, model, num_ctx, tools, outcome, strings)
        if belt.touched and not outcome.escalated:
            verdict = await asyncio.to_thread(gate_mod.run, cfg, sorted(belt.touched))
            outcome.gate_summary = verdict.summary()
            if verdict.ok and outcome.finished:
                outcome.ok = True
            else:
                outcome.gate_failures = verdict.failures()
                restored = belt.rollback()
                outcome.escalated = True
                outcome.reason = strings.verify_failed.format(
                    retry="", count=len(restored)
                )

    if not read_only and not belt.touched and not outcome.escalated:
        outcome.ok = False
        outcome.escalated = True
        outcome.reason = outcome.reason or strings.no_changes

    # The receipt states in as many words that an escalation leaves the working
    # tree untouched, and CLAUDE.md policies tell Claude to act on that. So it has
    # to hold on *every* escalation path -- including a model that edited files
    # and only then decided to hand the task back, which previously skipped the
    # gate block entirely and left those edits in place under an ESCALATE header.
    if outcome.escalated and belt.touched:
        restored = belt.rollback()
        if restored:
            outcome.reason = (
                f"{outcome.reason}; {strings.rolled_back.format(count=len(restored))}"
            ).lstrip("; ")

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
            local_tokens_estimated=outcome.local_tokens_estimated,
            elided_results=outcome.elided_results,
            receipt_chars=len(outcome.answer) + len(outcome.reason) + 200,
            # ~50 tokens covers the fixed scaffolding of a receipt: the header,
            # the diffstat lines and the gate summary.
            receipt_tokens_estimated=(
                encoding.estimate_tokens(outcome.answer)
                + encoding.estimate_tokens(outcome.reason)
                + 50
            ),
            gate=outcome.gate_summary,
        ),
    )
    return outcome


def _settle(outcome: Outcome, verdict: Verdict, body: str, strings: Strings) -> Outcome:
    """Record a verdict. Only DONE counts as success; everything else hands back."""
    if verdict is Verdict.DONE:
        outcome.finished = True
        outcome.answer = body
    else:
        outcome.escalated = True
        outcome.reason = body or strings.unclear_verdict
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
    strings: Strings,
) -> Outcome:
    clarified = False
    for _ in range(cfg.limits.max_iterations):
        outcome.iterations += 1
        outcome.elided_results += _fit_context(messages, tools, num_ctx)
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

        if calls:
            messages.append(
                {"role": "assistant", "content": result.content, "tool_calls": calls}
            )
            for call in calls:
                fn = call.get("function", {}) or {}
                name = fn.get("name", "")
                args = localtools.parse_args(fn.get("arguments"))
                if name == localtools.FINISH:
                    # The verdict arrived on the structured channel. Stop here --
                    # finish has no result to feed back, and anything the model
                    # queued after it is not part of the task.
                    belt.calls.append(name)
                    return _settle(
                        outcome,
                        sentinels.from_status(args.get("status")),
                        str(args.get("summary") or "").strip(),
                        strings,
                    )
                # to_thread: the belt reads files and can shell out to ripgrep
                # for up to a minute. Run inline it blocks the stdio server's
                # event loop, and Claude issues tool calls in parallel.
                tool_result = await asyncio.to_thread(belt.run, name, args)
                outcome.local_chars_consumed += len(tool_result)
                outcome.local_tokens_estimated += encoding.estimate_tokens(tool_result)
                messages.append({"role": "tool", "name": name, "content": tool_result})
            continue

        # No tool call: fall back to reading a verdict out of the prose.
        reply = sentinels.parse(result.content)
        if reply.verdict is not Verdict.UNKNOWN:
            return _settle(outcome, reply.verdict, reply.body or result.content.strip(), strings)

        # Unreadable. Ask once for a verdict in a form we can trust, then fail
        # safe. Treating "no ESCALATE found" as success is what made an untouched
        # tree look like a finished task.
        if clarified:
            outcome.escalated = True
            outcome.reason = strings.unclear_verdict
            return outcome
        clarified = True
        messages.append({"role": "assistant", "content": result.content})
        messages.append({"role": "user", "content": strings.restate_verdict})

    outcome.escalated = True
    outcome.reason = f"local model hit the {cfg.limits.max_iterations}-step limit without finishing"
    return outcome
