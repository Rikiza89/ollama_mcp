"""Savings accounting.

Without this you cannot tell whether delegation is saving tokens or merely
feeling like it. Every delegated call appends one JSONL row recording how much
text the local model consumed locally (which is what Claude did *not* have to
read) versus how large the receipt returned to Claude was.

`tokens_avoided` is a deliberate estimate -- bytes/4 is the usual rough ratio
for code -- and it is labelled as an estimate everywhere it surfaces.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Config

CHARS_PER_TOKEN = 4.0


@dataclass
class Record:
    tool: str
    model: str
    ok: bool
    escalated: bool
    duration_ms: int
    iterations: int
    local_prompt_tokens: int
    local_completion_tokens: int
    local_chars_consumed: int
    receipt_chars: int
    gate: str = ""
    ts: float = field(default_factory=time.time)

    @property
    def tokens_avoided(self) -> int:
        gross = self.local_chars_consumed / CHARS_PER_TOKEN
        paid = self.receipt_chars / CHARS_PER_TOKEN
        return int(max(0.0, gross - paid))


def append(cfg: Config, record: Record) -> None:
    try:
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        row = asdict(record)
        row["tokens_avoided_estimate"] = record.tokens_avoided
        with (cfg.state_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        # Never let bookkeeping break a delegation.
        pass


def summarize(cfg: Config, limit: int = 500) -> dict[str, Any]:
    path = cfg.state_dir / "metrics.jsonl"
    if not path.is_file():
        return {"calls": 0, "note": "no delegations recorded yet"}

    rows: list[dict[str, Any]] = []
    for line in _tail(path, limit):
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    if not rows:
        return {"calls": 0, "note": "no delegations recorded yet"}

    ok = sum(1 for r in rows if r.get("ok"))
    escalated = sum(1 for r in rows if r.get("escalated"))
    avoided = sum(int(r.get("tokens_avoided_estimate") or 0) for r in rows)
    paid = sum(int((r.get("receipt_chars") or 0) / CHARS_PER_TOKEN) for r in rows)
    durations = sorted(int(r.get("duration_ms") or 0) for r in rows)
    return {
        "calls": len(rows),
        "succeeded": ok,
        "escalated": escalated,
        "estimated_tokens_avoided": avoided,
        "tokens_spent_on_receipts": paid,
        "median_duration_s": round(durations[len(durations) // 2] / 1000, 1),
    }


def _tail(path: Path, limit: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]
