"""Tests for recovering tool calls that a model printed as prose.

Every shape here was observed from a real local model, or is one step away from
one. The gating rule -- only names we actually offer -- is what keeps a model
quoting JSON from turning into an execution.
"""

from __future__ import annotations

from ollama_mcp.toolcalls import extract

KNOWN = {"read_file", "edit_file", "write_file", "grep", "list_files"}


def test_bare_json_object() -> None:
    text = 'Let me look: {"name": "grep", "arguments": {"pattern": "threshold"}}'
    assert extract(text, KNOWN) == [
        {"function": {"name": "grep", "arguments": {"pattern": "threshold"}}}
    ]


def test_json_fence() -> None:
    text = 'I will read it.\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```'
    assert extract(text, KNOWN)[0]["function"]["name"] == "read_file"


def test_tool_call_tags() -> None:
    text = '<tool_call>\n{"name": "list_files", "arguments": {}}\n</tool_call>'
    assert extract(text, KNOWN)[0]["function"]["name"] == "list_files"


def test_already_native_shape() -> None:
    text = '{"function": {"name": "read_file", "arguments": {"path": "b.py"}}}'
    assert extract(text, KNOWN)[0]["function"]["arguments"] == {"path": "b.py"}


def test_arguments_inlined_next_to_name() -> None:
    text = '{"name": "read_file", "path": "c.py", "start_line": 3}'
    assert extract(text, KNOWN) == [
        {"function": {"name": "read_file", "arguments": {"path": "c.py", "start_line": 3}}}
    ]


def test_arguments_as_json_string() -> None:
    text = '{"name": "grep", "arguments": "{\\"pattern\\": \\"x\\"}"}'
    assert extract(text, KNOWN)[0]["function"]["arguments"] == {"pattern": "x"}


def test_braces_inside_argument_values_survive() -> None:
    """Tool arguments routinely contain code, so brace depth must be tracked."""
    text = (
        '{"name": "edit_file", "arguments": {"path": "a.py", '
        '"old_text": "def f(): {}", "new_text": "def f(): {\\"k\\": 1}"}}'
    )
    call = extract(text, KNOWN)[0]["function"]
    assert call["name"] == "edit_file"
    assert call["arguments"]["new_text"] == 'def f(): {"k": 1}'


def test_multiple_calls_in_order() -> None:
    text = (
        '{"name": "read_file", "arguments": {"path": "a.py"}} then '
        '{"name": "read_file", "arguments": {"path": "b.py"}}'
    )
    paths = [c["function"]["arguments"]["path"] for c in extract(text, KNOWN)]
    assert paths == ["a.py", "b.py"]


def test_duplicate_renderings_are_deduplicated() -> None:
    """A fenced call is also found by the bare scan; it must not run twice."""
    text = '```json\n{"name": "list_files", "arguments": {}}\n```'
    assert len(extract(text, KNOWN)) == 1


def test_unknown_tool_name_is_ignored() -> None:
    assert extract('{"name": "rm_rf", "arguments": {"path": "/"}}', KNOWN) == []


def test_prose_json_is_not_executed() -> None:
    text = 'The config looks like {"timeout": 30, "retries": 2} which seems fine.'
    assert extract(text, KNOWN) == []


def test_malformed_json_is_ignored() -> None:
    assert extract('{"name": "grep", "arguments": {', KNOWN) == []


def test_empty_content() -> None:
    assert extract("", KNOWN) == []
