"""Language resolution, prompts and receipt prose.

Two audiences, and they want opposite things:

* The **local model** must be prompted in the language it will be asked to work
  in. A qwen model given a Japanese instruction under an English system prompt
  answers in Japanese anyway, but drifts on the parts of the contract that
  matter -- it starts explaining instead of calling `edit_file`, and it
  translates the `DONE:` keyword. Prompting in Japanese fixes both.
* The **orchestrator** reading the receipt wants the status tokens to stay
  exactly where they were. `ESCALATE`, `APPLIED`, `PASS`, `FAIL` are protocol,
  not prose: `CLAUDE.md` files in the wild say "a tool returning ESCALATE means
  the working tree is unchanged", and a localized token silently breaks that
  contract. So only the explanatory sentences around them are translated.

`auto` (the default) decides per call: a task written in Japanese gets the
Japanese prompt regardless of what the machine's locale says, because the
language of the work is a better signal than the language of the OS.
"""

from __future__ import annotations

import locale
import os
import re
from dataclasses import dataclass
from enum import Enum

AUTO = "auto"

# Enough CJK to recognise Japanese prose. Kanji alone would also match Chinese,
# which is fine -- there is no Chinese prompt to mis-select into.
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
_JA_LOCALE = re.compile(r"^ja(?:[_\-.@]|$)|japanese", re.IGNORECASE)


class Language(str, Enum):
    EN = "en"
    JA = "ja"


def normalize_setting(value: object) -> str:
    """Validate a configured language setting at load time.

    Raises:
        TypeError: if the value is not a string at all.
        ValueError: if it is a string but not `auto`, `en` or `ja`. The caller
            wraps both in the config error type, so the server fails at startup
            with the offending file named rather than three delegations later.
    """
    if not isinstance(value, str):
        raise TypeError(f"language must be a string, got {type(value).__name__}")
    candidate = value.strip().lower()
    if candidate == AUTO or candidate in {lang.value for lang in Language}:
        return candidate
    allowed = ", ".join([AUTO, *(lang.value for lang in Language)])
    raise ValueError(f"unknown language {value!r}; expected one of: {allowed}")


def resolve(setting: str, *, hint: str = "") -> Language:
    """Turn a setting into a concrete language.

    Args:
        setting: `auto`, `en` or `ja`, already normalized.
        hint: The task text, when there is one. Under `auto` this wins over the
            machine locale.
    """
    if setting in {lang.value for lang in Language}:
        return Language(setting)
    if hint and _CJK.search(hint):
        return Language.JA
    return detect_from_locale() or Language.EN


def detect_from_locale() -> Language | None:
    """Best-effort read of the OS language. None when it says nothing useful."""
    for name in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        value = os.environ.get(name)
        if value and _JA_LOCALE.search(value.split(":")[0]):
            return Language.JA
    try:
        current = locale.getlocale()[0] or ""
    except ValueError:  # pragma: no cover - malformed platform locale
        current = ""
    return Language.JA if _JA_LOCALE.search(current) else None


EDIT_SYSTEM_EN = """You are a local code-editing assistant working inside a real repository.

Rules, in order of importance:
1. Do exactly what the task says. Do not refactor, rename, reformat, or "improve"
   anything you were not asked to change.
2. Always read a file before editing it. Copy `old_text` for edit_file verbatim from
   what you read, including indentation.
3. Prefer edit_file over write_file. Use write_file only for new files.
4. Keep the surrounding style: same naming, same comment density, same idioms.
5. Preserve the language and encoding of text already in the file. If a comment or
   docstring is in Japanese, leave it in Japanese unless the task says otherwise.
   Never convert a file's character encoding or its line endings.
6. When you are done, reply with plain text only -- no tool call -- in this shape:
   DONE: <one sentence on what you changed>
   If the task is ambiguous, impossible, or needs judgement you are not sure about,
   reply instead with:
   ESCALATE: <one sentence on exactly what is blocking you>
   Escalating is a correct outcome, not a failure. Never guess.
   Write the sentence in the same language as the task, but keep the leading
   `DONE:` / `ESCALATE:` keyword in ASCII exactly as shown."""

EDIT_SYSTEM_JA = """あなたは実際のリポジトリの中で作業する、ローカルのコード編集アシスタントです。

重要な順に、次の規則に従ってください。
1. 指示されたことだけを行う。頼まれていないリファクタリング、リネーム、整形、
   「改善」は一切しない。
2. 編集する前に必ずファイルを読む。edit_file の old_text は、読んだ内容から
   インデントも含めて一字一句そのまま写す。
3. write_file より edit_file を優先する。write_file は新規ファイルの作成にのみ使う。
4. 周囲のスタイルを保つ。命名、コメントの量、書き方を既存のコードに合わせる。
5. ファイルに既にある文章の言語と文字コードを保持する。コメントや docstring が
   日本語なら、指示がない限り日本語のまま残す。文字コードや改行コードを変換しない。
6. 終わったらツールを呼ばず、プレーンテキストで次の形だけを返す。
   DONE: <何を変更したかを一文で>
   指示があいまいな場合、実行できない場合、自信のない判断が必要な場合は、代わりに
   次を返す。
   ESCALATE: <何が障害になっているかを一文で>
   エスカレーションは失敗ではなく正しい結果です。推測で進めないこと。
   説明の文は指示と同じ言語で書いてよいが、先頭の `DONE:` / `ESCALATE:` は
   必ずここに示したとおり半角英字のまま書くこと。"""

READ_SYSTEM_EN = """You are a local code-reading assistant working inside a real repository.

Rules:
1. Answer only from what you actually read with the tools. Never invent file names,
   symbols, or line numbers.
2. Read narrowly: grep first, then read only the relevant line ranges.
3. You must not modify anything. You have no write tools.
4. Answer in the same language the question was asked in.
5. When done, reply with plain text only -- no tool call -- starting with:
   DONE: <your answer>
   Be dense and specific. Cite paths as path:line. Keep it under {budget} characters.
   If you cannot answer from the repository, reply:
   ESCALATE: <what is missing>
   Keep the leading `DONE:` / `ESCALATE:` keyword in ASCII exactly as shown."""

READ_SYSTEM_JA = """あなたは実際のリポジトリの中で作業する、ローカルのコード読解アシスタントです。

規則:
1. ツールで実際に読んだ内容だけから答える。ファイル名、シンボル名、行番号を
   推測で作らない。
2. 狭く読む。まず grep し、必要な行範囲だけを read_file する。
3. 何も変更してはならない。書き込み系のツールは渡されていない。
4. 質問された言語と同じ言語で答える。
5. 終わったらツールを呼ばず、プレーンテキストで次の形から始める。
   DONE: <答え>
   具体的かつ簡潔に。パスは path:line の形で示す。{budget} 文字以内に収める。
   リポジトリから答えられない場合は次を返す。
   ESCALATE: <何が不足しているか>
   先頭の `DONE:` / `ESCALATE:` は必ずここに示したとおり半角英字のまま書くこと。"""


@dataclass(frozen=True)
class Strings:
    """Localized prose. Status tokens are deliberately absent -- they never vary."""

    edit_system: str
    read_system: str
    steps: str
    tokens_read_locally: str
    verifier_said: str
    tree_unchanged: str
    gate_retry: str
    no_edit_nudge: str
    verify_failed: str
    after_retry: str
    no_changes: str
    triage_instruction: str
    no_gate_configured: str
    ollama_unavailable: str
    model_missing: str
    model_ok: str
    label_config: str
    label_models: str
    label_gate: str
    label_language: str
    label_savings: str
    gate_syntax_only: str


_EN = Strings(
    edit_system=EDIT_SYSTEM_EN,
    read_system=READ_SYSTEM_EN,
    steps="steps",
    tokens_read_locally="tok of file text read locally (est.)",
    verifier_said="verifier said:",
    tree_unchanged="-> The working tree is unchanged. Handle this one yourself.",
    gate_retry=(
        "Your change failed verification. Fix it, or reply ESCALATE if you cannot.\n\n"
    ),
    no_edit_nudge=(
        "You reported DONE but the working tree is unchanged -- you never called "
        "edit_file or write_file. Describing the change is not making it. Make the "
        "edit now with the tools, or reply ESCALATE with the reason you cannot."
    ),
    verify_failed="verification failed{retry}; rolled back {count} file(s)",
    after_retry=" after one local retry",
    no_changes="local model made no changes",
    triage_instruction=(
        "The project's checks failed. Below is the raw output. List each distinct "
        "problem as one line: `path:line - what is wrong - suggested fix`. Group "
        "duplicates. Do not use any tools unless you need to read a file to "
        "understand an error.\n\n"
    ),
    no_gate_configured=(
        "No gate configured and nothing autodetected.\nAdd a [gate] section to {path}."
    ),
    ollama_unavailable="Start Ollama, or do this work yourself.",
    model_missing="MISSING (run: ollama pull {model})",
    model_ok="ok",
    label_config="config",
    label_models="models",
    label_gate="gate",
    label_language="language",
    label_savings="savings so far",
    gate_syntax_only="(syntax checks only)",
)

_JA = Strings(
    edit_system=EDIT_SYSTEM_JA,
    read_system=READ_SYSTEM_JA,
    steps="ステップ",
    tokens_read_locally="トークン相当のファイル本文をローカルで読み込み(推定)",
    verifier_said="検証ツールの出力:",
    tree_unchanged="-> 作業ツリーは変更されていません。これは自分で対応してください。",
    gate_retry=(
        "あなたの変更は検証に失敗しました。修正するか、できない場合は ESCALATE と"
        "返してください。\n\n"
    ),
    no_edit_nudge=(
        "DONE と報告されましたが、作業ツリーは変更されていません。edit_file も "
        "write_file も一度も呼ばれていません。変更を説明することは変更することでは"
        "ありません。今すぐツールで編集するか、できない理由を添えて ESCALATE と"
        "返してください。"
    ),
    verify_failed="検証に失敗しました{retry}。{count} 件のファイルをロールバックしました",
    after_retry="(ローカルで1回再試行後)",
    no_changes="ローカルモデルは何も変更しませんでした",
    triage_instruction=(
        "プロジェクトのチェックが失敗しました。以下は生の出力です。個別の問題ごとに "
        "`path:line - 何が問題か - 修正案` の形式で一行ずつ列挙してください。"
        "重複はまとめること。エラーの理解にファイルの参照が必要な場合を除き、"
        "ツールは使わないでください。\n\n"
    ),
    no_gate_configured=(
        "ゲートが設定されておらず、自動検出もできませんでした。\n"
        "{path} に [gate] セクションを追加してください。"
    ),
    ollama_unavailable="Ollama を起動するか、この作業は自分で行ってください。",
    model_missing="未取得 (実行してください: ollama pull {model})",
    model_ok="ok",
    label_config="設定",
    label_models="モデル",
    label_gate="ゲート",
    label_language="言語",
    label_savings="これまでの削減量",
    gate_syntax_only="(構文チェックのみ)",
)

_STRINGS = {Language.EN: _EN, Language.JA: _JA}


def strings(language: Language) -> Strings:
    return _STRINGS[language]
