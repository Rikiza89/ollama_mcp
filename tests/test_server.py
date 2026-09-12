"""Receipt-discipline tests.

The saving comes entirely from what does NOT cross back into Claude's context,
so the receipt shape is the thing worth pinning down in tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from fake_ollama import FakeOllama, assistant, call

from ollama_mcp import config, server


async def test_tool_belt_stays_narrow() -> None:
    tools = await server.mcp.list_tools()
    assert {t.name for t in tools} == {
        "local_edit",
        "local_explain",
        "local_verify",
        "local_status",
    }


async def test_tool_schemas_stay_cheap() -> None:
    """These schemas are re-sent on every request forever. Keep them under budget."""
    tools = await server.mcp.list_tools()
    chars = sum(len(t.description or "") + len(json.dumps(t.input_schema)) for t in tools)
    assert chars // 4 < 1500, f"tool schemas cost ~{chars // 4} tokens per request"


async def test_no_tool_accepts_file_content() -> None:
    """Paths in, receipts out. A content parameter means Claude already read the file."""
    for tool in await server.mcp.list_tools():
        params = set(tool.input_schema.get("properties", {}))
        assert not params & {"content", "file_content", "code", "text"}, tool.name


async def test_edit_receipt_carries_no_file_body(tmp_path: Path, monkeypatch) -> None:
    secret = "SUPER_DISTINCTIVE_TOKEN_THAT_MUST_NOT_LEAK"
    (tmp_path / "a.py").write_text(f"x = 1  # {secret}\n", encoding="utf-8")
    (tmp_path / ".ollama-mcp.toml").write_text(
        "[gate]\ncommands = []\nautodetect = false\n", encoding="utf-8"
    )

    script = [
        assistant(tool_calls=[call("read_file", path="a.py")]),
        assistant(tool_calls=[call("edit_file", path="a.py", old_text="x = 1", new_text="x = 2")]),
        assistant("DONE: bumped x"),
    ]
    with FakeOllama(script) as fake:
        monkeypatch.setenv("OLLAMA_HOST_URL", fake.url)
        receipt = await server.local_edit(str(tmp_path), "set x to 2")

    assert "APPLIED" in receipt
    assert secret not in receipt
    assert "x = 2" not in receipt
    assert "+1/-1" in receipt
    assert len(receipt) < 500


async def test_escalation_receipt_tells_claude_to_take_over(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".ollama-mcp.toml").write_text(
        "[gate]\ncommands = []\nautodetect = false\n", encoding="utf-8"
    )
    with FakeOllama([assistant("ESCALATE: needs a design decision")]) as fake:
        monkeypatch.setenv("OLLAMA_HOST_URL", fake.url)
        receipt = await server.local_edit(str(tmp_path), "redesign the module")

    assert "ESCALATE" in receipt
    assert "working tree is unchanged" in receipt
    assert "Handle this one yourself" in receipt


async def test_status_reports_missing_models(tmp_path: Path, monkeypatch) -> None:
    with FakeOllama([], models=["qwen2.5-coder:7b"]) as fake:
        monkeypatch.setenv("OLLAMA_HOST_URL", fake.url)
        out = await server.local_status(str(tmp_path))

    assert "fast: qwen2.5-coder:7b -- ok" in out
    assert "MISSING (run: ollama pull qwen3-coder:30b)" in out


async def test_status_is_actionable_when_ollama_is_down(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST_URL", "http://127.0.0.1:1")
    out = await server.local_status(str(tmp_path))
    assert out.startswith("UNAVAILABLE")
    assert "do this work yourself" in out


def test_config_env_override_wins(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".ollama-mcp.toml").write_text(
        '[models]\nfast = "from-file"\n', encoding="utf-8"
    )
    assert config.load(tmp_path).models.fast == "from-file"
    monkeypatch.setenv("OLLAMA_MCP_FAST_MODEL", "from-env")
    assert config.load(tmp_path).models.fast == "from-env"
