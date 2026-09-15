"""Shared fixtures.

The important one is `_neutral_environment`. Almost every assertion in this
suite reads English prose, which means it silently depends on `i18n.AUTO`
resolving to English -- and `auto` consults the machine locale. On a Japanese
Windows install `locale.getlocale()` returns `('Japanese_Japan', '932')`, so
eight tests failed with correct Japanese output, on precisely the machine this
project exists to support. CI runs under the C locale and cannot catch it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from ollama_mcp import i18n

# Environment that leaks into `config.load` and would otherwise make results
# depend on the developer's shell rather than on the test.
_LEAKY_VARS = (
    "LC_ALL",
    "LC_MESSAGES",
    "LANG",
    "LANGUAGE",
    "OLLAMA_MCP_LANG",
    "OLLAMA_MCP_FAST_MODEL",
    "OLLAMA_MCP_DEEP_MODEL",
    "OLLAMA_HOST",
    "OLLAMA_HOST_URL",
)


@pytest.fixture(autouse=True)
def _neutral_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin language detection to "says nothing", so `auto` means English.

    Autouse and deliberately blunt. A test that wants a particular locale sets
    it afterwards -- monkeypatch applies in order, so the test's own setenv or
    setattr wins over this one.
    """
    for name in _LEAKY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(i18n.locale, "getlocale", lambda *args, **kwargs: (None, None))
