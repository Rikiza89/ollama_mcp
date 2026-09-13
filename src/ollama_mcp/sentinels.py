"""Reading the local model's final verdict out of its last message.

The loop ends when the model stops calling tools and replies in prose. That
prose has to answer one binary question -- did it finish, or is it handing the
task back? -- and getting it wrong in the ESCALATE direction is the worst
failure this server has: the working tree is unchanged, the receipt says
success, and the orchestrator moves on.

Matching `text.upper().startswith("ESCALATE")` gets that wrong the moment the
task is written in Japanese, because the model answers in the language it was
asked in: `エスカレート: 対象のファイルが見つかりません` was being reported as
an *answer*. So the keyword set is bilingual, the separator set includes the
fullwidth colon, and the usual markdown dressing (`**DONE:**`, a fenced line) is
stripped before matching.

The verdict now arrives through the `finish` tool -- a structured field on the
native tool-calling channel, kept apart from the model's prose. Everything below
is the fallback for models that ignore that channel, which plenty do.

Both routes share one rule, and it is the important one: **only an explicit
success marker means success.** Absence of a failure marker is not evidence that
the task went well. Reading it the other way round -- "no ESCALATE, therefore
APPLIED" -- puts the cost of every misread on the dangerous side: an untouched or
half-edited tree reported as done. Read this way, a misread costs one needless
"do it yourself", and the tree is guaranteed intact either way.

Credit for pinning down the direction of that asymmetry: @Skillselion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Leading noise a model wraps its verdict in: markdown emphasis, fences, block
# quotes, headings, and the Japanese quotation brackets.
_LEAD = re.compile(r'^[\s　>#*_`"\'\[\(「『【]+')

# What may sit between the keyword and the reason: closing emphasis or
# brackets and a separator, in either order -- `**DONE:** x` puts the
# separator inside the emphasis, `DONE**: x` outside. The fullwidth colon
# is the one a Japanese IME produces.
_CLOSER = r'[\s　]*[*_`\]\)」』】]*[\s　]*'
_SEPARATOR = r'[:：\-–—]*'
_AFTER = re.compile('^' + _CLOSER + _SEPARATOR + _CLOSER)

_ESCALATE = ("ESCALATE", "ESCALATION", "エスカレーション", "エスカレート")
_DONE = ("DONE", "COMPLETED", "COMPLETE", "完了")


class Verdict(Enum):
    DONE = "done"
    ESCALATE = "escalate"
    UNKNOWN = "unknown"


# Values the `finish` tool's status field may use for success. Anything else --
# a typo, an invented status, a missing field -- is an escalation by default.
_STATUS_DONE = frozenset({"done", "complete", "completed", "success", "ok", "完了"})


def from_status(raw: object) -> Verdict:
    """Map the `finish` tool's `status` argument onto a verdict.

    Only a recognised success value returns DONE. A missing field, a non-string,
    or a status the model invented all return ESCALATE -- the structured channel
    exists so that the safe direction is the default one, not so that we trust
    whatever turns up in it.
    """
    if not isinstance(raw, str):
        return Verdict.ESCALATE
    return Verdict.DONE if raw.strip().lower() in _STATUS_DONE else Verdict.ESCALATE


@dataclass(frozen=True)
class Reply:
    """A parsed final message.

    `body` is the reason or the answer with the keyword and separator removed,
    and is empty when the model sent a bare keyword. For `UNKNOWN` it is the
    message verbatim, because an unrecognised reply is still the best answer we
    have for a read-only task.
    """

    verdict: Verdict
    body: str

    @property
    def escalated(self) -> bool:
        return self.verdict is Verdict.ESCALATE


def parse(text: str) -> Reply:
    """Classify the model's closing message. Never raises."""
    raw = (text or "").strip()
    if not raw:
        return Reply(Verdict.UNKNOWN, "")

    head = _LEAD.sub("", raw)
    for verdict, keywords in ((Verdict.ESCALATE, _ESCALATE), (Verdict.DONE, _DONE)):
        for keyword in keywords:
            body = _match(head, keyword)
            if body is not None:
                return Reply(verdict, body)
    return Reply(Verdict.UNKNOWN, raw)


def _match(head: str, keyword: str) -> str | None:
    """Return the text after `keyword`, or None if `head` does not start with it.

    A bare `DONE` with nothing after it counts. `DONEISH` does not: the keyword
    has to be followed by end-of-string or something separator-shaped, or any
    word starting with those eight letters would end the loop.
    """
    if head[: len(keyword)].upper() != keyword.upper():
        return None
    rest = head[len(keyword) :]
    if not rest:
        return ""
    consumed = _AFTER.match(rest)
    if consumed is None or consumed.end() == 0:
        return None
    return rest[consumed.end() :].strip()
