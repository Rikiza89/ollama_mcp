"""Everything that crosses the bytes/text boundary, in one place.

Three of these boundaries behave differently on a Japanese machine than on the
CI runner, and each was its own bug:

* **Subprocess output.** `subprocess.run(text=True)` decodes with
  `locale.getpreferredencoding(False)` -- cp932 on a Japanese Windows install.
  ruff, pytest and ripgrep emit UTF-8 regardless of the console codepage, so the
  first non-ASCII byte raised `UnicodeDecodeError` out of the gate, where
  nothing caught it and the whole delegation died.
* **Source files with a BOM.** Japanese Windows editors write UTF-8 with a BOM
  by default. Handed to `ast.parse` as already-decoded *str*, that leading
  U+FEFF is an unconditional `SyntaxError` -- a false gate failure, so a correct
  edit gets rolled back and escalated. Handed over as *bytes*, the tokenizer
  strips it and honours a PEP 263 coding cookie too.
* **Token estimates.** chars/4 is an English-source ratio. A CJK character is
  closer to one token apiece, so both the savings figure and the local model's
  context budget were off by roughly 3x on a Japanese codebase.
"""

from __future__ import annotations

import locale
import re
import sys

UTF8_BOM = b"\xef\xbb\xbf"

# Kana, CJK ideographs (incl. extension A), compatibility forms and the
# fullwidth block. Deliberately not a full Unicode script test: this feeds an
# estimate, and the cost of being slightly wrong at the edges is nil.
_CJK = re.compile(
    r"[　-ヿ㐀-䶿一-鿿豈-﫿︰-﹏＀-￯]"
)

# Rough tokenizer ratios, in characters per token.
_ASCII_CHARS_PER_TOKEN = 4.0
_CJK_CHARS_PER_TOKEN = 1.0
_OTHER_CHARS_PER_TOKEN = 2.0


def decode_output(raw: bytes | None) -> str:
    """Decode captured subprocess output without ever raising.

    UTF-8 first because that is what modern dev tooling emits on every platform,
    then the console/locale encodings for the rare tool that really does honour
    the Windows codepage, then UTF-8 with replacement as a guaranteed floor.
    """
    if not raw:
        return ""
    for name in _candidate_encodings():
        try:
            return raw.decode(name)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _candidate_encodings() -> tuple[str, ...]:
    names = ["utf-8"]
    for guess in (locale.getpreferredencoding(False), sys.getfilesystemencoding()):
        if guess and guess.lower().replace("-", "") not in {"utf8", "ascii", "ansix341968"}:
            names.append(guess)
    if sys.platform == "win32":
        # The console codepage, which is not necessarily the ANSI one: cp932 on
        # a Japanese install, cp437 on a US one.
        names.append("oem")
    seen: set[str] = set()
    return tuple(n for n in names if not (n.lower() in seen or seen.add(n.lower())))


def strip_bom(raw: bytes) -> bytes:
    """Drop a leading UTF-8 BOM.

    `json.loads` autodetects UTF-8/16/32 from bytes per RFC 8259 but still
    rejects a UTF-8 BOM, which is exactly what a Japanese Windows editor leaves
    on a `package.json` or a config file.
    """
    return raw.removeprefix(UTF8_BOM)


def estimate_tokens(text: str) -> int:
    """Estimate the token count of `text` for a modern BPE tokenizer.

    An estimate, and labelled as one everywhere it surfaces. The point is not
    precision, it is that CJK text is not four characters to a token: counting
    it as if it were makes a Japanese repository look like it saved a third of
    what it actually did, and makes a "20k character" tool result three times
    the context the local model was budgeted.
    """
    if not text:
        return 0
    total = len(text)
    ascii_n = len(text.encode("ascii", "ignore"))
    cjk_n = len(_CJK.findall(text))
    other_n = max(0, total - ascii_n - cjk_n)
    return int(
        ascii_n / _ASCII_CHARS_PER_TOKEN
        + cjk_n / _CJK_CHARS_PER_TOKEN
        + other_n / _OTHER_CHARS_PER_TOKEN
    )


def byte_length(text: str) -> int:
    """UTF-8 byte length. `len(str)` counts characters, which understates a
    Japanese file by ~3x -- misleading in a receipt that says "bytes"."""
    return len(text.encode("utf-8", errors="surrogateescape"))
