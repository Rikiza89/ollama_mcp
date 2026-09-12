"""Japanese-environment behaviour, end to end.

Each test here corresponds to something that was broken before: a delegation
written in Japanese, a file a Japanese editor saved, a filename that came from a
Mac, and a receipt whose numbers were wrong by ~3x.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from fake_ollama import FakeOllama, assistant, call

from ollama_mcp import agent, config, gate, i18n, sandbox, server


def _project(tmp_path: Path, body: str, name: str = "a.py") -> Path:
    (tmp_path / name).write_text(body, encoding="utf-8")
    (tmp_path / ".ollama-mcp.toml").write_text(
        "[gate]\ncommands = []\nautodetect = false\n", encoding="utf-8"
    )
    return tmp_path


# -- the delegation itself -------------------------------------------------


async def test_a_japanese_instruction_gets_the_japanese_prompt(tmp_path: Path) -> None:
    root = _project(tmp_path, "def f():\n    return 1\n")
    script = [assistant("DONE: 何もしていません")]
    with FakeOllama(script) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        await agent.run_task(
            cfg,
            tool_name="local_explain",
            instruction="f 関数はどこで定義されていますか",
            tier="fast",
            read_only=True,
        )
    system = fake.requests[0]["messages"][0]["content"]
    assert "ローカルのコード読解アシスタント" in system
    # The keyword stays ASCII even in the Japanese prompt.
    assert "DONE:" in system and "ESCALATE:" in system


async def test_an_english_instruction_still_gets_the_english_prompt(tmp_path: Path) -> None:
    root = _project(tmp_path, "def f():\n    return 1\n")
    with FakeOllama([assistant("DONE: nothing")]) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        await agent.run_task(
            cfg, tool_name="local_explain", instruction="where is f defined?",
            tier="fast", read_only=True,
        )
    assert "local code-reading assistant" in fake.requests[0]["messages"][0]["content"]


async def test_configured_language_overrides_the_instruction(tmp_path: Path) -> None:
    root = _project(tmp_path, "def f():\n    return 1\n")
    (root / ".ollama-mcp.toml").write_text(
        '[gate]\ncommands = []\nautodetect = false\n\n[i18n]\nlanguage = "en"\n',
        encoding="utf-8",
    )
    with FakeOllama([assistant("DONE: nothing")]) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        await agent.run_task(
            cfg, tool_name="local_explain", instruction="f 関数はどこですか",
            tier="fast", read_only=True,
        )
    assert "local code-reading assistant" in fake.requests[0]["messages"][0]["content"]


async def test_a_japanese_escalation_is_an_escalation_not_an_answer(tmp_path: Path) -> None:
    """The regression this whole module exists for.

    `startswith("ESCALATE")` classified this as a successful answer, so the
    orchestrator was told the task was done over an unchanged working tree.
    """
    root = _project(tmp_path, "def f():\n    return 1\n")
    script = [assistant("エスカレート: 対象のファイルが見つかりません")]
    with FakeOllama(script) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        outcome = await agent.run_task(
            cfg, tool_name="local_edit", instruction="a.py の戻り値を 2 にして",
            tier="fast", read_only=False,
        )

    assert outcome.escalated
    assert not outcome.ok
    assert outcome.reason == "対象のファイルが見つかりません"
    assert "ESCALATE:" in server._receipt(outcome, header="NOT APPLIED")


async def test_japanese_comments_survive_a_delegated_edit(tmp_path: Path) -> None:
    body = "# 元のコメント\ndef f():\n    return 1\n"
    root = _project(tmp_path, body)
    script = [
        assistant(
            tool_calls=[
                call(
                    "edit_file",
                    path="a.py",
                    old_text="def f():",
                    new_text='def f():\n    """一行を返す。"""',
                )
            ]
        ),
        assistant("DONE: docstring を追加しました"),
    ]
    with FakeOllama(script) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        outcome = await agent.run_task(
            cfg, tool_name="local_edit", instruction="a.py に docstring を追加して",
            tier="fast", read_only=False,
        )

    assert outcome.ok
    assert outcome.answer == "docstring を追加しました"
    text = (root / "a.py").read_text(encoding="utf-8")
    assert "# 元のコメント" in text
    assert '"""一行を返す。"""' in text


async def test_the_receipt_is_localized_but_the_status_tokens_are_not(tmp_path: Path) -> None:
    root = _project(tmp_path, "def f():\n    return 1\n")
    script = [
        assistant(tool_calls=[call("read_file", path="a.py")]),
        assistant("エスカレート: 判断できません"),
    ]
    with FakeOllama(script) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        outcome = await agent.run_task(
            cfg, tool_name="local_edit", instruction="a.py を日本語化して",
            tier="fast", read_only=False,
        )

    receipt = server._receipt(outcome, header="NOT APPLIED")
    assert outcome.language is i18n.Language.JA
    assert "ステップ" in receipt
    assert "作業ツリーは変更されていません" in receipt
    # Protocol tokens: never translated.
    assert "NOT APPLIED" in receipt
    assert "ESCALATE:" in receipt
    assert receipt.startswith("NOT APPLIED")


async def test_tokens_read_locally_are_counted_per_script(tmp_path: Path) -> None:
    root = _project(tmp_path, "# " + "日" * 400 + "\ndef f():\n    return 1\n")
    script = [
        assistant(tool_calls=[call("read_file", path="a.py")]),
        assistant("DONE: 読みました"),
    ]
    with FakeOllama(script) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        outcome = await agent.run_task(
            cfg, tool_name="local_explain", instruction="a.py の内容を教えて",
            tier="fast", read_only=True,
        )

    # chars/4 would have reported roughly a quarter of this.
    assert outcome.local_tokens_estimated > outcome.local_chars_consumed // 2


# -- files a Japanese environment produces ---------------------------------


def test_a_bom_does_not_fail_the_syntax_gate(tmp_path: Path) -> None:
    root = _project(tmp_path, "x = 1\n")
    target = root / "bom.py"
    target.write_bytes(b"\xef\xbb\xbf# \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\nx = 1\n")
    assert gate.run(config.load(root), [target]).ok


def test_a_shift_jis_source_file_does_not_fail_the_syntax_gate(tmp_path: Path) -> None:
    root = _project(tmp_path, "x = 1\n")
    target = root / "sjis.py"
    target.write_bytes(
        "# -*- coding: cp932 -*-\n# 日本語のコメント\nx = 1\n".encode("cp932")
    )
    assert gate.run(config.load(root), [target]).ok


def test_a_bom_does_not_fail_the_json_gate(tmp_path: Path) -> None:
    root = _project(tmp_path, "x = 1\n")
    target = root / "data.json"
    target.write_bytes(b"\xef\xbb\xbf" + '{"名前": "値"}'.encode())
    assert gate.run(config.load(root), [target]).ok


def test_broken_python_still_fails_with_a_bom_present(tmp_path: Path) -> None:
    root = _project(tmp_path, "x = 1\n")
    target = root / "bad.py"
    target.write_bytes(b"\xef\xbb\xbf" + b"def f(:\n")
    assert not gate.run(config.load(root), [target]).ok


def test_utf8_command_output_does_not_crash_the_gate(tmp_path: Path) -> None:
    """`text=True` used to decode this with cp932 and raise out of the gate."""
    (tmp_path / ".ollama-mcp.toml").write_text(
        "[gate]\ncommands = [["
        '"python3", "-c", '
        "\"import sys; sys.stdout.buffer.write('検証エラー'.encode()); sys.exit(1)\""
        "]]\nautodetect = false\n",
        encoding="utf-8",
    )
    result = gate.run(config.load(tmp_path), [])
    assert not result.ok
    assert "検証エラー" in result.failures()


# -- filenames -------------------------------------------------------------


def test_a_decomposed_filename_is_found_by_its_composed_name(tmp_path: Path) -> None:
    """A repo authored on macOS, checked out on Linux, addressed by a model."""
    cfg = config.load(tmp_path)
    composed = "読み込み.py"
    decomposed = unicodedata.normalize("NFD", composed)
    (tmp_path / decomposed).write_text("x = 1\n", encoding="utf-8")

    resolved = sandbox.resolve(cfg, composed, must_exist=True)
    assert resolved.read_text(encoding="utf-8") == "x = 1\n"


def test_a_new_japanese_file_keeps_the_spelling_it_was_given(tmp_path: Path) -> None:
    cfg = config.load(tmp_path)
    assert sandbox.resolve(cfg, "新規ファイル.py").name == "新規ファイル.py"


def test_deny_rules_match_regardless_of_normalization(tmp_path: Path) -> None:
    (tmp_path / ".ollama-mcp.toml").write_text(
        '[sandbox]\ndeny = ["設定"]\n', encoding="utf-8"
    )
    cfg = config.load(tmp_path)
    decomposed = unicodedata.normalize("NFD", "設定")
    (tmp_path / decomposed).mkdir()
    (tmp_path / decomposed / "secret.py").write_text("x = 1\n", encoding="utf-8")
    try:
        sandbox.resolve(cfg, "設定/secret.py", must_exist=True)
        raise AssertionError("deny rule did not fire")
    except sandbox.SandboxError as exc:
        assert "設定" in str(exc)


def test_japanese_paths_still_cannot_escape_the_workspace(tmp_path: Path) -> None:
    cfg = config.load(tmp_path)
    try:
        sandbox.resolve(cfg, "../外部/秘密.py")
        raise AssertionError("path escaped the workspace")
    except sandbox.SandboxError as exc:
        assert "escapes workspace root" in str(exc)
