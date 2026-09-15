"""Qwen3-Coder emits tool calls in its own XML syntax, not JSON.

The sample below is copied verbatim from qwen3-coder:30b answering the real
edit system prompt with the real tool schemas attached. `tool_calls` came back
null, so without this the server saw a model that replied without doing
anything -- an escalation for a call the model had made correctly.
"""

from __future__ import annotations

from ollama_mcp import localtools, toolcalls

NL = chr(10)

KNOWN = {t["function"]["name"] for t in localtools.schemas()}

OBSERVED = "I'll call the read_file function to examine the specified section.\n\n<function=read_file>\n<parameter=path>\nservice/feature_gate.py\n</parameter>\n<parameter=start_line>\n45\n</parameter>\n<parameter=end_line>\n60\n</parameter>\n</function>\n</tool_call>"


def test_qwen_xml_tool_call_is_recovered() -> None:
    calls = toolcalls.extract(OBSERVED, KNOWN)
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "read_file"
    assert fn["arguments"] == {
        "path": "service/feature_gate.py",
        "start_line": "45",
        "end_line": "60",
    }


def test_recovered_numbers_are_coerced_by_the_schema() -> None:
    """The text channel has no types; the schema does."""
    args = localtools.coerce_args("read_file", {"path": "a.py", "start_line": "45"})
    assert args == {"path": "a.py", "start_line": 45}
    # A string parameter that merely looks numeric is left alone.
    assert localtools.coerce_args("write_file", {"path": "a", "content": "42"})["content"] == "42"


def test_multiline_values_keep_their_indentation() -> None:
    payload = NL.join([
        "<function=write_file>",
        "<parameter=path>",
        "a.py",
        "</parameter>",
        "<parameter=content>",
        "def f():",
        "    return 1",
        "</parameter>",
        "</function>",
    ])
    calls = toolcalls.extract(payload, KNOWN)
    assert calls[0]["function"]["arguments"]["content"] == "def f():" + NL + "    return 1"


def test_an_unknown_function_name_is_not_executed() -> None:
    payload = "<function=launch_missiles>" + NL + "<parameter=x>" + NL + "1" + NL + "</parameter>" + NL + "</function>"
    assert toolcalls.extract(payload, KNOWN) == []


def test_a_truncated_block_still_yields_the_call() -> None:
    """A long argument can run into the token limit before </function>."""
    payload = NL.join([
        "<function=edit_file>",
        "<parameter=path>",
        "a.py",
        "</parameter>",
    ])
    calls = toolcalls.extract(payload, KNOWN)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "edit_file"


def test_json_shapes_still_work() -> None:
    fenced = '```json' + NL + '{"name": "grep", "arguments": {"pattern": "x"}}' + NL + '```'
    calls = toolcalls.extract(fenced, KNOWN)
    assert calls[0]["function"]["name"] == "grep"
