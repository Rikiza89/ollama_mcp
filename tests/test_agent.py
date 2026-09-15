"""End-to-end agent-loop tests against the fake Ollama fixture.

These cover the behaviours the whole design rests on: a good edit is applied, a
bad edit is rolled back and escalated, and the receipt never carries file bodies.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fake_ollama import FakeOllama, assistant, call

from ollama_mcp import agent, config, metrics


def _project(tmp_path: Path, body: str = "def f():\n    return 1\n") -> Path:
    (tmp_path / "a.py").write_text(body, encoding="utf-8")
    # An explicit empty gate keeps autodetect from running the host's real ruff.
    (tmp_path / ".ollama-mcp.toml").write_text(
        "[gate]\ncommands = []\nautodetect = false\n", encoding="utf-8"
    )
    return tmp_path


async def test_successful_edit_is_applied(tmp_path: Path) -> None:
    root = _project(tmp_path)
    script = [
        assistant(tool_calls=[call("read_file", path="a.py")]),
        assistant(
            tool_calls=[call("edit_file", path="a.py", old_text="return 1", new_text="return 2")]
        ),
        assistant("DONE: changed the return value"),
    ]
    with FakeOllama(script) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        outcome = await agent.run_task(
            cfg, tool_name="local_edit", instruction="make f return 2", tier="fast",
            read_only=False,
        )

    assert outcome.ok and not outcome.escalated
    assert outcome.answer == "changed the return value"
    assert outcome.files == [{"path": "a.py", "status": "modified", "added": 1, "removed": 1}]
    assert (root / "a.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


async def test_syntax_failure_is_rolled_back_and_escalated(tmp_path: Path) -> None:
    root = _project(tmp_path)
    original = (root / "a.py").read_text(encoding="utf-8")
    broken = [
        assistant(
            tool_calls=[
                call("edit_file", path="a.py", old_text="return 1", new_text="return (((")
            ]
        ),
        assistant("DONE: done"),
    ]
    # Two rounds: the initial attempt and the one permitted retry, both broken.
    with FakeOllama(broken * 2) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        outcome = await agent.run_task(
            cfg, tool_name="local_edit", instruction="break it", tier="fast", read_only=False
        )

    assert not outcome.ok
    assert outcome.escalated
    assert "rolled back" in outcome.reason
    assert (root / "a.py").read_text(encoding="utf-8") == original


async def test_retry_recovers_from_a_bad_first_attempt(tmp_path: Path) -> None:
    root = _project(tmp_path)
    script = [
        assistant(
            tool_calls=[call("edit_file", path="a.py", old_text="return 1", new_text="return (((")]
        ),
        assistant("DONE: first attempt"),
        assistant(
            tool_calls=[
                call("edit_file", path="a.py", old_text="return (((", new_text="return 2")
            ]
        ),
        assistant("DONE: fixed after verification failure"),
    ]
    with FakeOllama(script) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        outcome = await agent.run_task(
            cfg, tool_name="local_edit", instruction="make f return 2", tier="fast",
            read_only=False,
        )

    assert outcome.ok
    assert (root / "a.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


async def test_model_escalation_leaves_tree_untouched(tmp_path: Path) -> None:
    root = _project(tmp_path)
    with FakeOllama([assistant("ESCALATE: the task is ambiguous")]) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        outcome = await agent.run_task(
            cfg, tool_name="local_edit", instruction="do something", tier="fast", read_only=False
        )

    assert outcome.escalated and not outcome.ok
    assert outcome.reason == "the task is ambiguous"
    assert (root / "a.py").read_text(encoding="utf-8") == "def f():\n    return 1\n"


async def test_no_changes_is_an_escalation_not_a_success(tmp_path: Path) -> None:
    root = _project(tmp_path)
    with FakeOllama([assistant("DONE: I decided nothing needed changing")]) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        outcome = await agent.run_task(
            cfg, tool_name="local_edit", instruction="rename x to y", tier="fast", read_only=False
        )

    assert outcome.escalated
    assert "no changes" in outcome.reason


async def test_iteration_cap_stops_a_looping_model(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / ".ollama-mcp.toml").write_text(
        "[gate]\ncommands = []\nautodetect = false\n\n[limits]\nmax_iterations = 3\n",
        encoding="utf-8",
    )
    looping = [assistant(tool_calls=[call("read_file", path="a.py")])] * 10
    with FakeOllama(looping) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        outcome = await agent.run_task(
            cfg, tool_name="local_edit", instruction="loop forever", tier="fast", read_only=False
        )

    assert outcome.escalated
    assert outcome.iterations == 3
    assert "step limit" in outcome.reason


async def test_read_only_task_gets_no_write_tools(tmp_path: Path) -> None:
    root = _project(tmp_path)
    with FakeOllama([assistant("DONE: f returns 1 (a.py:2)")]) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        outcome = await agent.run_task(
            cfg, tool_name="local_explain", instruction="what does f return?", tier="fast",
            read_only=True,
        )
        offered = {
            t["function"]["name"] for t in fake.requests[0]["tools"]
        }

    assert outcome.ok
    assert outcome.answer == "f returns 1 (a.py:2)"
    assert "edit_file" not in offered and "write_file" not in offered
    assert "read_file" in offered


async def test_num_ctx_is_always_sent(tmp_path: Path) -> None:
    """Ollama defaults num_ctx to 4096 and truncates silently. Never inherit it."""
    root = _project(tmp_path)
    with FakeOllama([assistant("DONE: ok")]) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        await agent.run_task(
            cfg, tool_name="local_explain", instruction="q", tier="deep", read_only=True
        )

    options = fake.requests[0]["options"]
    assert options["num_ctx"] == cfg.models.deep_num_ctx
    assert fake.requests[0]["model"] == cfg.models.deep


async def test_metrics_row_is_written(tmp_path: Path) -> None:
    root = _project(tmp_path)
    with FakeOllama([assistant("DONE: an answer")]) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        await agent.run_task(
            cfg, tool_name="local_explain", instruction="q", tier="fast", read_only=True
        )

    summary = metrics.summarize(cfg)
    assert summary["calls"] == 1
    assert summary["succeeded"] == 1


@pytest.mark.parametrize("tier,expected", [("fast", "fast"), ("deep", "deep")])
def test_pick_model_respects_tier(tmp_path: Path, tier: str, expected: str) -> None:
    cfg = config.load(tmp_path)
    name, num_ctx = agent.pick_model(cfg, tier)
    assert name == getattr(cfg.models, expected)
    assert num_ctx == getattr(cfg.models, f"{expected}_num_ctx")


# --- the transcript gets a budget, not just each result ---------------------


def _tool_message(size: int) -> dict:
    return {"role": "tool", "name": "read_file", "content": "x" * size}


def test_a_short_transcript_is_left_alone() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _tool_message(400),
    ]
    assert agent._fit_context(messages, [], num_ctx=16384) == 0
    assert messages[2]["content"] == "x" * 400


def test_old_tool_results_are_dropped_to_fit_the_window() -> None:
    """Capping each result does not cap their sum.

    Twelve iterations of a 5,000-token result is 60,000 tokens going into a
    16,384-token window, where Ollama truncates silently and the model edits a
    file it never fully saw.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        *[_tool_message(20_000) for _ in range(12)],
    ]
    dropped = agent._fit_context(messages, [], num_ctx=16384)

    assert dropped > 0
    # Oldest first, and the most recent result is the one kept.
    assert messages[2]["content"] == agent.ELIDED
    assert messages[-1]["content"] == "x" * 20_000
    total = sum(len(str(m["content"])) for m in messages)
    assert total / 4 < 16384


def test_the_task_itself_is_never_dropped() -> None:
    messages = [
        {"role": "system", "content": "s" * 40_000},
        {"role": "user", "content": "t" * 40_000},
        _tool_message(40_000),
    ]
    agent._fit_context(messages, [], num_ctx=4096)
    assert messages[0]["content"] == "s" * 40_000
    assert messages[1]["content"] == "t" * 40_000


def test_fitting_is_idempotent() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        *[_tool_message(20_000) for _ in range(12)],
    ]
    agent._fit_context(messages, [], num_ctx=16384)
    assert agent._fit_context(messages, [], num_ctx=16384) == 0


def test_tool_schemas_count_against_the_budget() -> None:
    from ollama_mcp import localtools

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _tool_message(12_000),
    ]
    bare = agent._fit_context([dict(m) for m in messages], [], num_ctx=4096)
    with_tools = agent._fit_context(messages, localtools.schemas(), num_ctx=4096)
    assert with_tools >= bare
