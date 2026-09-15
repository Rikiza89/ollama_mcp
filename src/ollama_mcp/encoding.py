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


def detect_text(raw: bytes) -> tuple[str, str]:
    """Decode source bytes losslessly, returning the text and the codec used.

    Everything the local model is *shown* and everything it *matches against*
    has to be the same string, or `edit_file` can never succeed: `read_file`
    used to decode with `errors="replace"`, so a cp932 source file reached the
    model as U+FFFD, and the `old_text` it copied back verbatim then failed to
    match the surrogateescape-decoded content on disk. The task burned its whole
    iteration budget before escalating.

    So: one decode, and one that round-trips. UTF-8 first (with the BOM spelling
    kept, so writing the file back restores it), then the machine's own
    encoding -- cp932 on a Japanese Windows install, where a legacy Shift-JIS
    source file is an ordinary thing to find -- and finally latin-1, which is a
    bijection over all 256 byte values and therefore cannot fail. The last case
    shows the model mojibake, but the bytes it does not touch survive the edit
    unchanged, which is the property that actually matters.
    """
    if raw.startswith(UTF8_BOM):
        return raw[len(UTF8_BOM) :].decode("utf-8"), "utf-8-sig"
    for name in _source_encodings():
        try:
            return raw.decode(name), name
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1"), "latin-1"


def _source_encodings() -> tuple[str, ...]:
    """UTF-8, then whatever legacy encoding this machine's source files use.

    `getpreferredencoding` is not enough on its own: under UTF-8 mode (a
    `PYTHONUTF8=1` environment, or `python -X utf8`) it answers "utf-8" and
    hides the fact that the machine's own codepage is cp932 -- so a Shift-JIS
    source file fell through to the latin-1 floor and reached the model as
    mojibake. `locale.getencoding()` reports the real codepage regardless of
    UTF-8 mode; `getlocale()` supplies the same number on the versions that
    predate it.
    """
    names = ["utf-8"]
    for guess in (_locale_encoding(), _locale_codepage()):
        if guess and guess.lower().replace("-", "") not in {"utf8", "ascii", "ansix341968"}:
            names.append(guess)
    seen: set[str] = set()
    return tuple(n for n in names if not (n.lower() in seen or seen.add(n.lower())))


def _locale_encoding() -> str:
    # locale.getencoding() is 3.11+; it is the one that ignores UTF-8 mode.
    getencoding = getattr(locale, "getencoding", None)
    if getencoding is not None:
        return getencoding()
    return locale.getpreferredencoding(False)


def _locale_codepage() -> str:
    """The locale's codepage as a codec name, e.g. ('Japanese_Japan', '932') -> cp932."""
    try:
        number = (locale.getlocale()[1] or "").strip()
    except ValueError:  # pragma: no cover - malformed platform locale
        return ""
    return f"cp{number}" if number.isdigit() else number


class UnrepresentableText(ValueError):
    """The edited text cannot be written back in the file's own encoding."""


def encode_text(text: str, codec: str) -> bytes:
    """Re-encode with the codec `detect_text` reported, never silently upgrading.

    Converting the file to UTF-8 because the new text does not fit its existing
    encoding would rewrite every byte of a file the task never asked to touch --
    the system prompt tells the model in as many words not to do that. So a
    character the file's encoding cannot hold is an error handed back to the
    model, which escalates, rather than a conversion nobody asked for.
    """
    try:
        return text.encode(codec)
    except UnicodeEncodeError as exc:
        raise UnrepresentableText(
            f"the text contains characters that {codec} cannot represent "
            f"({exc.object[exc.start : exc.end]!r}); this file is {codec}-encoded "
            f"and must stay that way"
        ) from exc


def looks_binary(raw: bytes) -> bool:
    """A NUL byte in the first block. The usual heuristic, and the right one here.

    The previous test was "does this decode as strict UTF-8" -- which on a
    Japanese machine excluded every cp932 source file in the repository from
    `grep`, quietly, as though those files held no matches.
    """
    return b"\0" in raw[:8192]


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
