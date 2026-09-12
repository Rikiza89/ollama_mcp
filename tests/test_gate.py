from __future__ import annotations

from pathlib import Path

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


def test_missing_command_is_skipped_not_failed(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        '[gate]\ncommands = [["definitely-not-a-real-binary-xyz"]]\nautodetect = false\n',
    )
    result = gate.run(cfg, [])
    assert result.ok
    assert "not installed" in result.checks[0].output


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
