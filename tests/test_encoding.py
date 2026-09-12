"""Byte/text boundary behaviour, which is where the Japanese-environment bugs were."""

from __future__ import annotations

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
