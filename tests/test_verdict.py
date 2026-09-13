"""How a delegated task is allowed to end.

One rule, and every test here is a face of it: **only an explicit success marker
means success.** The absence of a failure marker proves nothing about how the
task went, and treating it as proof puts the cost of every misread on the
dangerous side -- an untouched or half-edited tree reported as APPLIED.

The direction of that asymmetry was pointed out by @Skillselion, on the writeup
of the Japanese-environment work. The original code read
`text.upper().startswith("ESCALATE")`, which says in as many words: no failure
marker, therefore success.

Two invariants follow, and both are asserted below:
  1. APPLIED requires a verdict the server could actually read.
  2. ESCALATE means the working tree is byte-for-byte unchanged -- on every
     escalation path, including a model that edited files and only then decided
     to hand the task back.
"""

from __future__ import annotations

from pathlib import Path

from fake_ollama import FakeOllama, assistant, call

from ollama_mcp import agent, config, server

ORIGINAL = "def f():\n    return 1\n"
EDIT = call("edit_file", path="a.py", old_text="return 1", new_text="return 2")


def _project(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text(ORIGINAL, encoding="utf-8")
    (tmp_path / ".ollama-mcp.toml").write_text(
        "[gate]\ncommands = []\nautodetect = false\n", encoding="utf-8"
    )
    return tmp_path


async def _run(root: Path, script: list, *, read_only: bool = False) -> agent.Outcome:
    with FakeOllama(script) as fake:
        cfg = config.load(root)
        cfg.ollama_host = fake.url
        return await agent.run_task(
            cfg,
            tool_name="local_edit",
            instruction="make f return 2",
            tier="fast",
            read_only=read_only,
        )


def _finish(status: str, summary: str = "x") -> dict:
    return call("finish", status=status, summary=summary)


# -- the structured channel ------------------------------------------------


async def test_finish_done_applies(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outcome = await _run(root, [assistant(tool_calls=[EDIT]), assistant(tool_calls=[_finish("done", "returns 2 now")])])
    assert outcome.ok and outcome.finished and not outcome.escalated
    assert outcome.answer == "returns 2 now"
    assert (root / "a.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


async def test_finish_escalate_rolls_back(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outcome = await _run(root, [assistant(tool_calls=[EDIT]), assistant(tool_calls=[_finish("escalate", "not sure")])])
    assert outcome.escalated and not outcome.ok
    assert (root / "a.py").read_text(encoding="utf-8") == ORIGINAL
    assert outcome.files == []


async def test_an_invented_status_escalates(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outcome = await _run(root, [assistant(tool_calls=[EDIT]), assistant(tool_calls=[call("finish", status="probably", summary="?")])])
    assert outcome.escalated
    assert (root / "a.py").read_text(encoding="utf-8") == ORIGINAL


async def test_a_missing_status_escalates(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outcome = await _run(root, [assistant(tool_calls=[EDIT]), assistant(tool_calls=[call("finish", summary="?")])])
    assert outcome.escalated
    assert (root / "a.py").read_text(encoding="utf-8") == ORIGINAL


async def test_finish_printed_as_prose_is_still_honoured(tmp_path: Path) -> None:
    # Models that ignore the tool channel print the call instead; toolcalls.py
    # recovers it, and a recovered escalation must escalate like any other.
    root = _project(tmp_path)
    outcome = await _run(
        root,
        [
            assistant(tool_calls=[EDIT]),
            assistant('{"name": "finish", "arguments": {"status": "escalate", "summary": "無理でした"}}'),
        ],
    )
    assert outcome.escalated
    # The reason keeps the model's words and gains the rollback note.
    assert outcome.reason.startswith("無理でした")
    assert "rolled back 1" in outcome.reason
    assert (root / "a.py").read_text(encoding="utf-8") == ORIGINAL


async def test_work_queued_after_finish_is_ignored(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outcome = await _run(root, [assistant(tool_calls=[_finish("escalate", "stopping"), EDIT])])
    assert outcome.escalated
    assert (root / "a.py").read_text(encoding="utf-8") == ORIGINAL


# -- the prose fallback ----------------------------------------------------


async def test_an_escalation_with_a_sentence_in_front_of_it_escalates(tmp_path: Path) -> None:
    """@Skillselion's case.

    The model announces the escalation conversationally instead of leading with
    the keyword. Under `startswith` this was indistinguishable from success.
    """
    root = _project(tmp_path)
    outcome = await _run(
        root,
        [
            assistant(tool_calls=[EDIT]),
            assistant("承知しました。ESCALATE します"),
            assistant("ESCALATE: やはり判断できません"),
        ],
    )
    assert outcome.escalated and not outcome.ok
    assert (root / "a.py").read_text(encoding="utf-8") == ORIGINAL


async def test_an_unreadable_verdict_is_asked_once_then_escalates(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outcome = await _run(
        root,
        [assistant(tool_calls=[EDIT]), assistant("うーん"), assistant("たぶん大丈夫です")],
    )
    assert outcome.escalated and not outcome.ok
    assert (root / "a.py").read_text(encoding="utf-8") == ORIGINAL


async def test_the_clarifying_round_can_recover_a_real_success(tmp_path: Path) -> None:
    # The re-ask is what keeps the escalation rate honest: a model that simply
    # forgot the keyword gets one chance to say so plainly.
    root = _project(tmp_path)
    outcome = await _run(
        root,
        [assistant(tool_calls=[EDIT]), assistant("うーん"), assistant("DONE: changed it")],
    )
    assert outcome.ok and outcome.finished
    assert (root / "a.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


async def test_only_one_clarifying_round_is_spent(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outcome = await _run(root, [assistant("hm"), assistant("hmm"), assistant("DONE: too late")])
    assert outcome.escalated


# -- read-only -------------------------------------------------------------


async def test_a_refusal_is_not_returned_as_an_answer(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outcome = await _run(
        root,
        [assistant("すみません、判断できませんでした"), assistant("まだ分かりません")],
        read_only=True,
    )
    assert outcome.escalated and not outcome.ok


async def test_read_only_success_still_needs_an_explicit_verdict(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outcome = await _run(root, [assistant(tool_calls=[_finish("done", "a.py:1")])], read_only=True)
    assert outcome.ok and outcome.answer == "a.py:1"


# -- the receipt tells the truth -------------------------------------------


async def test_the_receipt_never_claims_an_untouched_tree_falsely(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outcome = await _run(root, [assistant(tool_calls=[EDIT]), assistant(tool_calls=[_finish("escalate", "handing back")])])
    receipt = server._receipt(outcome, header="NOT APPLIED")
    assert "ESCALATE:" in receipt
    assert (root / "a.py").read_text(encoding="utf-8") == ORIGINAL, (
        "the receipt says the tree is unchanged; it has to be true"
    )
    assert "  modified" not in receipt
