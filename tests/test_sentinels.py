"""Verdict parsing.

The case that matters most is the Japanese ESCALATE: misreading it as an answer
is the only way this server can report success over an unchanged working tree.
"""

from __future__ import annotations

import pytest

from ollama_mcp.sentinels import Verdict, parse


@pytest.mark.parametrize(
    ("text", "body"),
    [
        ("ESCALATE: file not found", "file not found"),
        ("escalate: lowercase works", "lowercase works"),
        ("エスカレート: 対象のファイルが見つかりません", "対象のファイルが見つかりません"),
        ("エスカレーション：判断できません", "判断できません"),
        ("ESCALATE：全角コロン", "全角コロン"),
        ("**ESCALATE:** wrapped in markdown", "wrapped in markdown"),
        ("「ESCALATE」: 括弧付き", "括弧付き"),
    ],
)
def test_escalations_are_recognised(text: str, body: str) -> None:
    reply = parse(text)
    assert reply.escalated
    assert reply.body == body


@pytest.mark.parametrize(
    ("text", "body"),
    [
        ("DONE: added docstrings", "added docstrings"),
        ("DONE：日本語のコメントを追加しました", "日本語のコメントを追加しました"),
        ("完了: 3件の docstring を追加", "3件の docstring を追加"),
        ("**DONE:** markdown", "markdown"),
        ("DONE**: markdown the other way", "markdown the other way"),
        ("## DONE: heading", "heading"),
        ("done - dash separator", "dash separator"),
    ],
)
def test_completions_are_recognised(text: str, body: str) -> None:
    reply = parse(text)
    assert reply.verdict is Verdict.DONE
    assert reply.body == body


def test_bare_keyword_has_an_empty_body() -> None:
    assert parse("DONE").verdict is Verdict.DONE
    assert parse("DONE").body == ""


def test_a_word_merely_starting_with_a_keyword_is_not_a_verdict() -> None:
    assert parse("DONEISH, kind of").verdict is Verdict.UNKNOWN
    assert parse("ESCALATED the issue upstream myself").verdict is Verdict.UNKNOWN


def test_prose_is_unknown_and_kept_verbatim() -> None:
    reply = parse("I rewrote the loop and it looks fine now.")
    assert reply.verdict is Verdict.UNKNOWN
    assert reply.body == "I rewrote the loop and it looks fine now."
    assert not reply.escalated


def test_empty_message() -> None:
    assert parse("").verdict is Verdict.UNKNOWN
    assert parse("   ").body == ""
