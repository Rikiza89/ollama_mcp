"""A model that edits and then never finishes must leave the tree untouched.

This is the exact shape seen in the field: qwen3-coder:30b rewrote a file, kept
working, and ran out of iterations. The receipt said NOT APPLIED and "the
working tree is unchanged" -- so that had better be true, because published
CLAUDE.md policies tell the orchestrator to act on it.
"""

from __future__ import annotations

from pathlib import Path

from fake_ollama import FakeOllama, assistant, call

from ollama_mcp import agent, config


def _project(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".ollama-mcp.toml").write_text(
        "[gate]\ncommands = []\nautodetect = false\n\n[limits]\nmax_iterations = 3\n",
        encoding="utf-8",
    )
    return tmp_path


async def test_edits_then_step_limit_leaves_the_tree_unchanged(tmp_path: Path) -> None:
    root = _project(tmp_path)
    original = (root / "a.py").read_bytes()

    # Edits on every turn, never calls finish -- runs straight into the limit.
    script = [
        assistant(tool_calls=[call("write_file", path="a.py", content="x = 999\n")]),
        assistant(tool_calls=[call("write_file", path="a.py", content="x = 888\n")]),
        assistant(tool_calls=[call("write_file", path="a.py", content="x = 777\n")]),
    ]
    with FakeOllama(script) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        outcome = await agent.run_task(
            cfg, tool_name="local_edit", instruction="translate it",
            tier="fast", read_only=False,
        )

    assert outcome.escalated
    assert not outcome.ok
    assert "step limit" in outcome.reason
    # The contract the receipt prints.
    assert (root / "a.py").read_bytes() == original, "tree was NOT restored"
    # And the receipt must not advertise changes it just rolled back.
    assert outcome.files == [], f"receipt still lists {outcome.files}"


async def test_edits_then_explicit_escalate_leaves_the_tree_unchanged(tmp_path: Path) -> None:
    root = _project(tmp_path)
    original = (root / "a.py").read_bytes()

    script = [
        assistant(tool_calls=[call("write_file", path="a.py", content="x = 999\n")]),
        assistant(tool_calls=[call("finish", status="escalate", summary="changed my mind")]),
    ]
    with FakeOllama(script) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        outcome = await agent.run_task(
            cfg, tool_name="local_edit", instruction="translate it",
            tier="fast", read_only=False,
        )

    assert outcome.escalated
    assert (root / "a.py").read_bytes() == original, "tree was NOT restored"
    assert outcome.files == []
