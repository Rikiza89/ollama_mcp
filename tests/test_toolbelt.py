from __future__ import annotations

from pathlib import Path

import pytest

from ollama_mcp import config
from ollama_mcp.localtools import ToolBelt, parse_args, schemas


@pytest.fixture()
def belt(tmp_path: Path) -> ToolBelt:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return ToolBelt(cfg=config.load(tmp_path))


def test_belt_is_small() -> None:
    # A wide belt degrades small models and costs them context. Hold the line.
    assert len(schemas()) <= 5


def test_read_file_numbers_lines(belt: ToolBelt) -> None:
    out = belt.run("read_file", {"path": "a.py"})
    assert "1  def f():" in out
    assert "lines 1-2 of 2" in out


def test_read_file_range(belt: ToolBelt) -> None:
    out = belt.run("read_file", {"path": "a.py", "start_line": 2, "end_line": 2})
    assert "return 1" in out
    assert "def f()" not in out


def test_edit_file_applies_and_tracks(belt: ToolBelt) -> None:
    assert belt.run("edit_file", {"path": "a.py", "old_text": "return 1", "new_text": "return 2"}
                    ).startswith("OK")
    assert (belt.cfg.workspace / "a.py").read_text(encoding="utf-8").endswith("return 2\n")
    stat = belt.diffstat()
    assert stat == [{"path": "a.py", "status": "modified", "added": 1, "removed": 1}]


def test_edit_file_refuses_ambiguous_match(belt: ToolBelt) -> None:
    (belt.cfg.workspace / "b.py").write_text("x\nx\n", encoding="utf-8")
    out = belt.run("edit_file", {"path": "b.py", "old_text": "x", "new_text": "y"})
    assert "appears 2 times" in out
    assert not belt.touched


def test_edit_file_reports_missing_anchor(belt: ToolBelt) -> None:
    out = belt.run("edit_file", {"path": "a.py", "old_text": "nope", "new_text": "y"})
    assert "not found" in out


def test_write_file_creates_and_rollback_deletes(belt: ToolBelt) -> None:
    belt.run("write_file", {"path": "new/c.py", "content": "z = 3\n"})
    created = belt.cfg.workspace / "new" / "c.py"
    assert created.is_file()
    assert belt.diffstat()[0]["status"] == "created"

    belt.rollback()
    assert not created.exists()


def test_rollback_restores_original_content(belt: ToolBelt) -> None:
    original = (belt.cfg.workspace / "a.py").read_text(encoding="utf-8")
    belt.run("edit_file", {"path": "a.py", "old_text": "return 1", "new_text": "return 999"})
    belt.rollback()
    assert (belt.cfg.workspace / "a.py").read_text(encoding="utf-8") == original


def test_read_only_belt_refuses_writes(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    ro = ToolBelt(cfg=config.load(tmp_path), read_only=True)
    assert "read-only" in ro.run("write_file", {"path": "a.py", "content": "bad"})
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\n"


def test_unknown_tool_is_reported_not_raised(belt: ToolBelt) -> None:
    assert belt.run("launch_missiles", {}).startswith("ERROR: unknown tool")


def test_sandbox_violation_returns_error_string(belt: ToolBelt) -> None:
    assert belt.run("read_file", {"path": "../../etc/passwd"}).startswith("ERROR:")


def test_parse_args_accepts_json_string() -> None:
    assert parse_args('{"path": "a.py"}') == {"path": "a.py"}
    assert parse_args("not json") == {}
    assert parse_args({"path": "b.py"}) == {"path": "b.py"}


def _raw(path: Path) -> bytes:
    return path.read_bytes()


def test_edit_preserves_lf_line_endings(tmp_path: Path) -> None:
    """On Windows, text-mode writes rewrite LF to CRLF and turn a one-line edit
    into a whole-file diff. Bytes outside the edit must not move."""
    target = tmp_path / "lf.py"
    target.write_bytes(b"a = 1\nb = 2\nc = 3\n")
    belt = ToolBelt(cfg=config.load(tmp_path))
    belt.run("edit_file", {"path": "lf.py", "old_text": "b = 2", "new_text": "b = 22"})
    assert _raw(target) == b"a = 1\nb = 22\nc = 3\n"


def test_edit_preserves_crlf_line_endings(tmp_path: Path) -> None:
    target = tmp_path / "crlf.py"
    target.write_bytes(b"a = 1\r\nb = 2\r\n")
    belt = ToolBelt(cfg=config.load(tmp_path))
    belt.run("edit_file", {"path": "crlf.py", "old_text": "b = 2", "new_text": "b = 22"})
    assert _raw(target) == b"a = 1\r\nb = 22\r\n"


def test_rollback_is_byte_identical(tmp_path: Path) -> None:
    target = tmp_path / "lf.py"
    original = b"x = 1\ny = 2\n"
    target.write_bytes(original)
    belt = ToolBelt(cfg=config.load(tmp_path))
    belt.run("edit_file", {"path": "lf.py", "old_text": "x = 1", "new_text": "x = 999"})
    belt.rollback()
    assert _raw(target) == original


def test_grep_is_smart_case(tmp_path: Path) -> None:
    """Lowercase pattern matches anything; an uppercase letter makes it exact."""
    (tmp_path / "s.py").write_text("FREE_THRESHOLD = 75\nother = 1\n", encoding="utf-8")
    belt = ToolBelt(cfg=config.load(tmp_path))
    assert "FREE_THRESHOLD" in belt.run("grep", {"pattern": "threshold"})
    assert belt.run("grep", {"pattern": "Threshold"}) == "no matches"
