"""Byte/text boundary behaviour, which is where the Japanese-environment bugs were."""

from __future__ import annotations

import pytest

from ollama_mcp import encoding


def test_utf8_subprocess_output_decodes() -> None:
    assert encoding.decode_output("ruff: 日本語エラー".encode()) == "ruff: 日本語エラー"


def test_decode_never_raises_on_undecodable_bytes() -> None:
    # The regression: `text=True` would raise UnicodeDecodeError here on a
    # cp932 console and take the whole delegation down with it.
    assert isinstance(encoding.decode_output(b"\x81\x40\xff\xfe garbage"), str)


def test_decode_handles_empty_and_none() -> None:
    assert encoding.decode_output(None) == ""
    assert encoding.decode_output(b"") == ""


def test_strip_bom_only_strips_a_leading_bom() -> None:
    assert encoding.strip_bom(encoding.UTF8_BOM + b'{"a": 1}') == b'{"a": 1}'
    assert encoding.strip_bom(b'{"a": 1}') == b'{"a": 1}'
    inner = b'{"a": ' + encoding.UTF8_BOM + b"1}"
    assert encoding.strip_bom(inner) == inner


def test_japanese_costs_far_more_tokens_per_character_than_ascii() -> None:
    # The whole point of the script-aware estimate: 100 characters of Japanese
    # is not 25 tokens, and reporting it as 25 undersells delegation by ~3x.
    assert encoding.estimate_tokens("a" * 100) == 25
    assert encoding.estimate_tokens("あ" * 100) >= 90


def test_mixed_script_is_counted_per_run() -> None:
    mixed = "def 読み込み():"
    assert encoding.estimate_tokens(mixed) > encoding.estimate_tokens("def readfile():")


def test_estimate_tokens_of_empty_string() -> None:
    assert encoding.estimate_tokens("") == 0


def test_byte_length_counts_utf8_bytes_not_characters() -> None:
    assert encoding.byte_length("あいう") == 9
    assert encoding.byte_length("abc") == 3


# --- one decode, and one that round-trips -----------------------------------


def test_detect_text_round_trips_a_cp932_source_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case the whole project is built around.

    `read_file` used to decode with errors="replace" while `edit_file` used
    surrogateescape, so the model was shown U+FFFD, copied it back verbatim as
    old_text, and could never match. The task burned its entire iteration
    budget before escalating.
    """
    raw = "# 設定を読み込む\nx = 1\n".encode("cp932")
    # Pinned rather than inherited: which legacy codec `detect_text` reaches
    # for depends on the machine, and CI does not run a Japanese locale.
    monkeypatch.setattr(encoding, "_locale_encoding", lambda: "cp932")
    text, codec = encoding.detect_text(raw)
    assert codec == "cp932"
    assert "設定" in text
    assert encoding.encode_text(text, codec) == raw


def test_detect_text_preserves_a_bom_exactly_once() -> None:
    raw = "# こんにちは\ny = 1\n".encode("utf-8-sig")
    text, codec = encoding.detect_text(raw)
    assert not text.startswith("\ufeff")
    assert encoding.encode_text(text, codec) == raw


def test_detect_text_never_fails() -> None:
    """latin-1 is a bijection over all 256 byte values, so it cannot raise."""
    raw = bytes(range(256))
    text, codec = encoding.detect_text(raw)
    assert encoding.encode_text(text, codec) == raw


def test_encode_text_refuses_to_convert_the_file() -> None:
    """Rule 5 of the system prompt: never convert a file's character encoding.

    A character the file's codec cannot hold is an error handed back to the
    model, which escalates -- not a silent upgrade to UTF-8 that rewrites every
    byte of a file the task only asked to add a comment to.
    """
    with pytest.raises(encoding.UnrepresentableText, match="cp932"):
        encoding.encode_text("x = 1  # ✓", "cp932")


def test_looks_binary_uses_nul_not_utf8_validity() -> None:
    # The old test was "does this decode as strict UTF-8", which excluded every
    # cp932 source file in the repository from grep.
    assert not encoding.looks_binary("# 設定".encode("cp932"))
    assert encoding.looks_binary(b"MZ\x00\x00payload")
