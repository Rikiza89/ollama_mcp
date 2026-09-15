"""Recovering tool calls that a local model emitted as prose.

Ollama advertises a `tools` capability for a model whenever its template knows
how to render tool schemas -- but plenty of models (qwen2.5-coder among them)
will happily ignore the native channel and print the call into the message body
instead, as bare JSON, inside a ```json fence, or wrapped in <tool_call> tags.

Left unhandled this looks exactly like "the model finished without doing
anything", which is the wrong diagnosis and produces a needless escalation. So
we parse the text as a fallback, and only when the native field is empty.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Qwen3-Coder's own tool-call syntax. Not JSON at all, so nothing above sees it:
#
#     <function=edit_file>
#     <parameter=path>
#     service/feature_gate.py
#     </parameter>
#     </function>
#
# The model emits this happily even when handed native tool schemas, and the
# result looked exactly like "the model answered without doing anything" -- a
# needless escalation, and on a long task a march straight into the step limit
# as it tried again and again.
_XML_FUNCTION = re.compile(r"<function=([A-Za-z_][A-Za-z_0-9]*)\s*>(.*?)</function>", re.DOTALL)
_XML_PARAMETER = re.compile(r"<parameter=([A-Za-z_][A-Za-z_0-9]*)\s*>(.*?)</parameter>", re.DOTALL)
# Same thing with the closing tag missing, which happens when a long argument
# runs into the token limit. Anchored to the end so it cannot swallow a
# well-formed block earlier in the message.
_XML_FUNCTION_OPEN = re.compile(r"<function=([A-Za-z_][A-Za-z_0-9]*)\s*>(?!.*</function>)(.*)\Z", re.DOTALL)
_FENCE = re.compile(r"```(?:json|tool_code|python)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract(content: str, known: set[str]) -> list[dict[str, Any]]:
    """Return native-shaped tool calls found in free text, in order of appearance.

    `known` gates the result: an object is only treated as a call if its name is
    a tool we actually offer. That keeps ordinary prose containing JSON -- a
    model quoting a config file, say -- from being executed.
    """
    if not content:
        return []

    calls: list[dict[str, Any]] = []
    seen: set[str] = set()

    # If the model chose the XML syntax, take it and stop. Scanning the same
    # message for JSON as well would treat a brace-y `content` argument -- a
    # config file, a dict literal, any code with braces in it -- as a second
    # call to run.
    xml = _xml_calls(content, known)
    if xml:
        return xml

    for candidate in _candidates(content):
        parsed = _parse(candidate, known)
        if parsed is None:
            continue
        fingerprint = json.dumps(parsed, sort_keys=True)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        calls.append(parsed)

    return calls


def _xml_calls(content: str, known: set[str]) -> list[dict[str, Any]]:
    """Read Qwen-style `<function=name><parameter=key>value</parameter></function>`."""
    out: list[dict[str, Any]] = []
    matched_spans: list[tuple[int, int]] = []

    for match in _XML_FUNCTION.finditer(content):
        matched_spans.append(match.span())
        call = _xml_one(match.group(1), match.group(2), known)
        if call is not None:
            out.append(call)

    if not out:
        truncated = _XML_FUNCTION_OPEN.search(content)
        if truncated and not any(a <= truncated.start() < b for a, b in matched_spans):
            call = _xml_one(truncated.group(1), truncated.group(2), known)
            if call is not None:
                out.append(call)
    return out


def _xml_one(name: str, body: str, known: set[str]) -> dict[str, Any] | None:
    if name not in known:
        return None
    arguments: dict[str, Any] = {}
    for parameter in _XML_PARAMETER.finditer(body):
        arguments[parameter.group(1)] = _xml_value(parameter.group(2))
    return {"function": {"name": name, "arguments": arguments}}


def _xml_value(raw: str) -> str:
    """Undo the one newline the format adds on each side of a value.

    `strip()` would be wrong: these values carry file content, where leading
    indentation and trailing blank lines are load-bearing.
    """
    for opener in ("\r\n", "\n"):
        if raw.startswith(opener):
            raw = raw[len(opener):]
            break
    for closer in ("\r\n", "\n"):
        if raw.endswith(closer):
            raw = raw[: -len(closer)]
            break
    return raw


def _candidates(content: str) -> list[str]:
    out = [m.group(1) for m in _TOOL_CALL_TAG.finditer(content)]
    out += [m.group(1) for m in _FENCE.finditer(content)]
    out += _balanced_objects(content)
    return out


def _balanced_objects(content: str) -> list[str]:
    """Scan for top-level {...} runs, respecting strings and escapes.

    A regex cannot do this: tool arguments routinely contain braces (code!) and
    quoted braces, so depth has to be tracked properly.
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, char in enumerate(content):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(content[start : index + 1])
    return out


def _parse(candidate: str, known: set[str]) -> dict[str, Any] | None:
    try:
        data = json.loads(candidate)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None

    # Shape 1: already native -- {"function": {"name": ..., "arguments": {...}}}
    function = data.get("function")
    if isinstance(function, dict):
        data = function

    name = data.get("name") or data.get("tool") or data.get("tool_name")
    if not isinstance(name, str) or name not in known:
        return None

    arguments = data.get("arguments")
    if arguments is None:
        arguments = data.get("parameters")
    if arguments is None:
        # Shape 2: arguments inlined alongside the name.
        arguments = {k: v for k, v in data.items() if k not in {"name", "tool", "tool_name"}}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if not isinstance(arguments, dict):
        return None

    return {"function": {"name": name, "arguments": arguments}}
