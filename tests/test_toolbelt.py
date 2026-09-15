from __future__ import annotations

from pathlib import Path

import pytest

from ollama_mcp import config
from ollama_mcp.localtools import FINISH, ToolBelt, parse_args, schemas


@pytest.fixture()
def belt(tmp_path: Path) -> ToolBelt:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return ToolBelt(cfg=config.load(tmp_path))


def test_belt_is_small() -> None:
    # A wide belt degrades small models and costs them context. Hold the line.
    # Five capability tools plus `finish`, which is protocol rather than
    # capability: it is what moves the verdict off the prose channel, and it
    # pays for its slot by removing the guesswork that channel required.
    assert len(schemas()) <= 6
    capability = [t["function"]["name"] for t in schemas() if t["function"]["name"] != FINISH]
    assert len(capability) <= 5


def test_finish_only_admits_the_two_verdicts() -> None:
    finish = next(t for t in schemas() if t["function"]["name"] == FINISH)
    params = finish["function"]["parameters"]
    assert params["properties"]["status"]["enum"] == ["done", "escalate"]
    assert set(params["required"]) == {"status", "summary"}


def test_finish_is_not_a_belt_handler(belt: ToolBelt) -> None:
    # The agent loop intercepts it; reaching the belt would mean the loop missed
    # the verdict and carried on as though the task were still running.
    assert not hasattr(belt, f"_t_{FINISH}")


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


# --- the deny list governs every route to a file's contents -----------------


@pytest.fixture()
def secrets(tmp_path: Path) -> ToolBelt:
    (tmp_path / "app.py").write_text("token = FREE\n", encoding="utf-8")
    (tmp_path / "deploy.pem").write_text("PRIVATE-KEY-BODY\n", encoding="utf-8")
    (tmp_path / "id_rsa").write_text("PRIVATE-KEY-BODY\n", encoding="utf-8")
    (tmp_path / "prod.env").write_text("API_KEY=PRIVATE-KEY-BODY\n", encoding="utf-8")
    return ToolBelt(cfg=config.load(tmp_path))


def test_grep_does_not_leak_what_read_file_refuses(secrets: ToolBelt) -> None:
    """The guard had two implementations and therefore one hole.

    `read_file` consulted the deny list properly; `grep` did its own part-only,
    lowercased comparison that never looked at extensions, so a key `read_file`
    would not open came back a line at a time through search.
    """
    assert secrets.run("read_file", {"path": "deploy.pem"}).startswith("ERROR:")
    hits = secrets.run("grep", {"pattern": "PRIVATE-KEY-BODY"})
    assert "PRIVATE-KEY-BODY" not in hits
    for denied in ("deploy.pem", "id_rsa", "prod.env"):
        assert denied not in hits


def test_list_files_does_not_name_denied_files(secrets: ToolBelt) -> None:
    listing = secrets.run("list_files", {})
    assert "app.py" in listing
    for denied in ("deploy.pem", "id_rsa", "prod.env"):
        assert denied not in listing


def test_grep_still_finds_ordinary_files(secrets: ToolBelt) -> None:
    assert "app.py" in secrets.run("grep", {"pattern": "FREE"})


# --- line windows ----------------------------------------------------------


def test_read_file_rejects_a_window_past_the_end(belt: ToolBelt) -> None:
    """This used to return a header reading "lines 99-2 of 2" and no error."""
    out = belt.run("read_file", {"path": "a.py", "start_line": 99})
    assert out.startswith("ERROR:")
    assert "only 2 lines" in out


def test_read_file_rejects_an_inverted_window(belt: ToolBelt) -> None:
    out = belt.run("read_file", {"path": "a.py", "start_line": 2, "end_line": 1})
    assert out.startswith("ERROR:")


# --- rollback leaves nothing behind ----------------------------------------


def test_rollback_removes_directories_it_created(belt: ToolBelt) -> None:
    belt.run("write_file", {"path": "new/deep/c.py", "content": "z = 3\n"})
    assert (belt.cfg.workspace / "new" / "deep").is_dir()
    belt.rollback()
    assert not (belt.cfg.workspace / "new").exists()


def test_rollback_keeps_directories_that_already_existed(belt: ToolBelt) -> None:
    (belt.cfg.workspace / "pkg").mkdir()
    belt.run("write_file", {"path": "pkg/c.py", "content": "z = 3\n"})
    belt.rollback()
    assert (belt.cfg.workspace / "pkg").is_dir()


# --- what the model is shown is what edit_file matches against ---------------


def test_a_legacy_encoded_file_can_actually_be_edited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this pair of methods existed to have.

    read_file decoded with errors="replace" and edit_file with surrogateescape,
    so the model was shown U+FFFD where the Japanese comment was, copied that
    back verbatim as old_text, and could never match. Every edit to a cp932
    file escalated after spending the full iteration budget.
    """
    monkeypatch.setattr("ollama_mcp.encoding._locale_encoding", lambda: "cp932")
    source = tmp_path / "s.py"
    source.write_bytes("# 設定を読み込む\nx = 1\n".encode("cp932"))
    belt = ToolBelt(cfg=config.load(tmp_path))

    shown = belt.run("read_file", {"path": "s.py"})
    assert "設定を読み込む" in shown
    # Exactly what a model copies out of that listing: the gutter is 5 wide
    # plus two spaces.
    copied = shown.splitlines()[1][7:]
    assert belt.run("edit_file", {"path": "s.py", "old_text": copied, "new_text": "# loaded"})

    assert source.read_bytes() == "# loaded\nx = 1\n".encode("cp932")


def test_an_edit_never_converts_the_files_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ollama_mcp.encoding._locale_encoding", lambda: "cp932")
    source = tmp_path / "s.py"
    original = "# 設定\nx = 1\n".encode("cp932")
    source.write_bytes(original)
    belt = ToolBelt(cfg=config.load(tmp_path))

    out = belt.run("edit_file", {"path": "s.py", "old_text": "x = 1", "new_text": "x = 1  # ✓"})
    assert out.startswith("ERROR:")
    assert "cp932" in out
    # Refused, not silently upgraded to UTF-8 -- which would have rewritten
    # every byte of a file the task only asked to append a comment to.
    assert source.read_bytes() == original


def test_a_bom_survives_an_edit(tmp_path: Path) -> None:
    source = tmp_path / "b.py"
    source.write_bytes("# こんにちは\ny = 1\n".encode("utf-8-sig"))
    belt = ToolBelt(cfg=config.load(tmp_path))

    belt.run("edit_file", {"path": "b.py", "old_text": "y = 1", "new_text": "y = 2"})
    assert source.read_bytes() == "# こんにちは\ny = 2\n".encode("utf-8-sig")


def test_rollback_restores_bytes_exactly(tmp_path: Path) -> None:
    source = tmp_path / "c.py"
    original = "a = 1\r\nb = 2\r\n".encode("utf-8-sig")
    source.write_bytes(original)
    belt = ToolBelt(cfg=config.load(tmp_path))

    belt.run("edit_file", {"path": "c.py", "old_text": "b = 2", "new_text": "b = 22"})
    belt.rollback()
    assert source.read_bytes() == original


# --- a full rewrite must not change the file's line endings ------------------


def test_write_file_keeps_crlf_endings(tmp_path: Path) -> None:
    """The documented guarantee was only half true.

    edit_file preserved a file's line endings; write_file wrote whatever the
    model produced, and a model emits LF. Flipping a CRLF file wholesale is a
    whole-file diff for a one-line change, and .gitattributes pins real paths
    to specific endings (`*.html -text`, `*.bat eol=crlf`).
    """
    source = tmp_path / "t.html"
    source.write_bytes(b"<p>uno</p>\r\n<p>due</p>\r\n")
    belt = ToolBelt(cfg=config.load(tmp_path))

    belt.run("write_file", {"path": "t.html", "content": "<p>one</p>\n<p>two</p>\n"})
    assert source.read_bytes() == b"<p>one</p>\r\n<p>two</p>\r\n"


def test_write_file_keeps_lf_endings(tmp_path: Path) -> None:
    source = tmp_path / "t.sh"
    source.write_bytes(b"echo uno\necho due\n")
    belt = ToolBelt(cfg=config.load(tmp_path))

    belt.run("write_file", {"path": "t.sh", "content": "echo one\r\necho two\r\n"})
    assert source.read_bytes() == b"echo one\necho two\n"


def test_a_new_file_keeps_what_the_model_wrote(tmp_path: Path) -> None:
    belt = ToolBelt(cfg=config.load(tmp_path))
    belt.run("write_file", {"path": "fresh.txt", "content": "a\nb\n"})
    assert (tmp_path / "fresh.txt").read_bytes() == b"a\nb\n"


def test_edit_file_matches_the_replacement_to_the_file(tmp_path: Path) -> None:
    source = tmp_path / "t.py"
    source.write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")
    belt = ToolBelt(cfg=config.load(tmp_path))

    # A multi-line replacement typed with LF, going into a CRLF file.
    belt.run("edit_file", {"path": "t.py", "old_text": "b = 2", "new_text": "b = 20\nbb = 21"})
    raw = source.read_bytes()
    assert raw == b"a = 1\r\nb = 20\r\nbb = 21\r\nc = 3\r\n"
    assert b"\n" not in raw.replace(b"\r\n", b"")


# --- disambiguating a repeated fragment -------------------------------------


@pytest.fixture()
def repeated(tmp_path: Path) -> ToolBelt:
    # "Scegli" on lines 1 and 4, with filler between.
    (tmp_path / "page.html").write_text(
        "<option>Scegli</option>\n<p>filler</p>\n<p>filler</p>\n<option>Scegli</option>\n",
        encoding="utf-8",
    )
    return ToolBelt(cfg=config.load(tmp_path))


def test_a_repeated_fragment_asks_for_near_line(repeated: ToolBelt) -> None:
    """The old advice -- add surrounding context -- is advice a small model
    cannot follow, because the surrounding context has to be reassembled from
    read_file's numbered output. Point it at something it already knows."""
    out = repeated.run("edit_file", {"path": "page.html", "old_text": "Scegli", "new_text": "Pick"})
    assert out.startswith("ERROR:")
    assert "appears 2 times" in out
    assert "near_line" in out


def test_near_line_picks_the_intended_occurrence(repeated: ToolBelt) -> None:
    out = repeated.run(
        "edit_file",
        {"path": "page.html", "old_text": "Scegli", "new_text": "Pick", "near_line": 4},
    )
    assert out.startswith("OK:")
    lines = (repeated.cfg.workspace / "page.html").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "<option>Scegli</option>"   # untouched
    assert lines[3] == "<option>Pick</option>"     # the one asked for


def test_near_line_picks_the_first_when_that_is_closest(repeated: ToolBelt) -> None:
    repeated.run(
        "edit_file",
        {"path": "page.html", "old_text": "Scegli", "new_text": "Pick", "near_line": 1},
    )
    lines = (repeated.cfg.workspace / "page.html").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "<option>Pick</option>"
    assert lines[3] == "<option>Scegli</option>"


def test_a_missing_fragment_explains_the_gutter(belt: ToolBelt) -> None:
    out = belt.run("edit_file", {"path": "a.py", "old_text": "    1  def f():", "new_text": "x"})
    assert out.startswith("ERROR:")
    assert "NOT part of the file" in out
    assert "Do not retry the same old_text" in out


def test_near_line_is_coerced_from_the_text_channel(repeated: ToolBelt) -> None:
    """Recovered XML tool calls deliver every argument as a string."""
    out = repeated.run(
        "edit_file",
        {"path": "page.html", "old_text": "Scegli", "new_text": "Pick", "near_line": "4"},
    )
    assert out.startswith("OK:")
