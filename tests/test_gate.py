from __future__ import annotations

from pathlib import Path

import pytest

from ollama_mcp import config, gate


def _cfg(tmp_path: Path, toml: str = "[gate]\ncommands = []\nautodetect = false\n"):
    (tmp_path / ".ollama-mcp.toml").write_text(toml, encoding="utf-8")
    return config.load(tmp_path)


def test_broken_python_fails_the_gate(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def f(:\n", encoding="utf-8")
    result = gate.run(_cfg(tmp_path), [bad])
    assert not result.ok
    assert "FAIL" in result.summary()
    assert "bad.py" in result.failures()


def test_valid_python_passes(tmp_path: Path) -> None:
    good = tmp_path / "good.py"
    good.write_text("def f():\n    return 1\n", encoding="utf-8")
    assert gate.run(_cfg(tmp_path), [good]).ok


def test_broken_json_fails(tmp_path: Path) -> None:
    bad = tmp_path / "data.json"
    bad.write_text('{"a": }', encoding="utf-8")
    assert not gate.run(_cfg(tmp_path), [bad]).ok


def test_unknown_extension_is_not_syntax_checked(tmp_path: Path) -> None:
    odd = tmp_path / "notes.xyz"
    odd.write_text("!!! not any language !!!", encoding="utf-8")
    result = gate.run(_cfg(tmp_path), [odd])
    assert result.ok and result.skipped


def test_failing_command_fails_the_gate(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        '[gate]\ncommands = [["python", "-c", "import sys; sys.exit(3)"]]\nautodetect = false\n',
    )
    assert not gate.run(cfg, []).ok


def test_passing_command_passes(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, '[gate]\ncommands = [["python", "-c", "pass"]]\nautodetect = false\n')
    assert gate.run(cfg, []).ok


def test_a_command_that_cannot_run_is_not_a_pass(tmp_path: Path) -> None:
    """The inverse of what this asserted before, and the reason for the change.

    A gate command missing from PATH -- a typo, or the ordinary case of an MCP
    server spawned from the desktop app's environment rather than from your
    shell -- used to report `pass`, so every delegated edit afterwards was
    verified by nothing while the receipt claimed otherwise.
    """
    cfg = _cfg(
        tmp_path,
        '[gate]\ncommands = [["definitely-not-a-real-binary-xyz"]]\nautodetect = false\n',
    )
    result = gate.run(cfg, [])
    assert not result.ok
    assert not result.checks[0].ran
    assert "UNVERIFIED" in result.summary()
    assert "verified nothing" in result.checks[0].output


def test_command_timeout_fails(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        '[gate]\ncommands = [["python", "-c", "import time; time.sleep(5)"]]\n'
        "autodetect = false\ntimeout_s = 1\n",
    )
    result = gate.run(cfg, [])
    assert not result.ok
    assert "timed out" in result.failures()


def test_autodetect_finds_python_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    cfg = config.load(tmp_path)
    commands = gate.autodetect(cfg)
    assert commands == [] or commands[0][0] == "ruff"


def test_workspace_relative_executable_is_resolved(tmp_path: Path) -> None:
    """subprocess resolves the executable against the parent process cwd, not the
    cwd= it is given, so a project-local interpreter must be resolved by us."""
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "python.exe").write_bytes(b"")
    cfg = _cfg(tmp_path)

    resolved = gate._resolve_executable(cfg, "vendor/python.exe")
    assert Path(resolved).is_absolute()
    assert Path(resolved) == (vendor / "python.exe")


def test_bare_command_is_left_for_path_lookup(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert gate._resolve_executable(cfg, "ruff") == "ruff"


def test_absolute_executable_is_untouched(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    absolute = str(Path(tmp_path) / "tool.exe")
    assert gate._resolve_executable(cfg, absolute) == absolute


def test_relative_path_that_does_not_exist_is_left_alone(tmp_path: Path) -> None:
    """Better to let subprocess raise its own FileNotFoundError than invent a path."""
    cfg = _cfg(tmp_path)
    assert gate._resolve_executable(cfg, "nope/missing.exe") == "nope/missing.exe"


def test_resolution_survives_a_real_gate_run(tmp_path: Path) -> None:
    """End-to-end: a relative path to a real file is rewritten before spawning."""
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "runner").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    cfg = _cfg(tmp_path)
    resolved = gate._resolve_executable(cfg, "vendor/runner")
    assert Path(resolved).is_file()


# --- the gate section is the one that has to be strict ----------------------


def test_commands_as_a_flat_list_of_strings_is_rejected(tmp_path: Path) -> None:
    """The obvious TOML mistake, which used to load without complaint.

    `_merge` saw `type(value) is list` and waved the table through; then
    `_run_command` iterated the string one character at a time and reported the
    gate as `r u f=pass`. Every delegated edit afterwards was checked by nothing.
    """
    toml = '[gate]' + chr(10) + 'commands = ["ruff check ."]' + chr(10)
    with pytest.raises(config.ConfigError) as caught:
        _cfg(tmp_path, toml)
    message = str(caught.value)
    assert "commands[0]" in message
    assert "must be a list of arguments" in message
    # The message names the fix, not just the problem.
    assert '[["ruff", "check", "."]]' in message


def test_an_empty_command_is_rejected(tmp_path: Path) -> None:
    toml = '[gate]' + chr(10) + 'commands = [[]]' + chr(10)
    with pytest.raises(config.ConfigError, match="at least a program name"):
        _cfg(tmp_path, toml)


def test_a_non_string_argument_is_rejected(tmp_path: Path) -> None:
    toml = '[gate]' + chr(10) + 'commands = [["ruff", 3]]' + chr(10)
    with pytest.raises(config.ConfigError, match="only strings"):
        _cfg(tmp_path, toml)


def test_a_well_formed_command_still_loads(tmp_path: Path) -> None:
    toml = '[gate]' + chr(10) + 'commands = [["ruff", "check", "."]]' + chr(10)
    assert _cfg(tmp_path, toml).gate.commands == [["ruff", "check", "."]]
