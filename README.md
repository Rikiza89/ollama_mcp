# ollama-mcp-delegate

**日本語版: [README.ja.md](README.ja.md)** · [日本語環境での挙動](docs/ja/japanese-environment.md)

An MCP server that lets **Claude Code stay the orchestrator** while a **local Ollama model does the mechanical work** — file edits, code lookups, lint triage — so that text never enters Claude's context and you pay fewer tokens for the same result.

```
  you ──▶ Claude Code (Anthropic)          ← decides, plans, reviews
              │
              │  paths + instruction        (small)
              ▼
        ollama-mcp-delegate  ──▶ Ollama ──▶ qwen2.5-coder / qwen3-coder
              │                      │
              │                      └── reads files, greps, edits — all locally
              │
              ▼  verification gate (syntax + your project's checks)
              │
              │  receipt: files changed, +12/-4, gate=pass   (small)
              ▼
        Claude Code
```

## The one idea

**Delegation must displace content, not add a hop.** The tools take *paths and an instruction*, never file contents, and return a *receipt*, never code. If Claude has to read a file in order to write the delegation prompt, the tokens are already spent and you have saved nothing.

The second idea, which is what makes the output trustworthy: **nothing is reported as success until it passes a deterministic gate.** Parity with pure Claude Code doesn't come from the local model being clever — it comes from a narrow task plus a machine-checkable result, with automatic rollback and escalation when the check fails.

## What it is not

- It does **not** replace Claude. Routing all of Claude Code at Ollama via `ANTHROPIC_BASE_URL` is a different thing — that removes Claude rather than creating a hybrid.
- It is **not** a way to make Claude Code subagents run locally. Subagent `model:` frontmatter accepts Anthropic tiers only; this is the supported seam instead.
- It will **not** help with architecture, debugging, or ambiguity. Expect roughly **30–60% savings on delegable classes and ~0% on the hard thinking.** Anyone promising more is counting wrong.

## Install

Requires Python 3.10+ and [Ollama](https://ollama.com). [ripgrep](https://github.com/BurntSushi/ripgrep) is used for search when present; there is a pure-Python fallback, which matters on Windows where `rg` often exists only inside Git Bash and not on the system PATH.

```bash
git clone https://github.com/Rikiza89/ollama_mcp
cd ollama_mcp
uv venv && uv pip install -e .

ollama pull qwen2.5-coder:7b     # fast tier
ollama pull qwen3-coder:30b      # deep tier (~18 GB)
```

Register it with Claude Code, once, for all your projects:

```bash
claude mcp add -s user ollama-local -- /abs/path/to/ollama_mcp/.venv/bin/ollama-mcp
# Windows: ...\ollama_mcp\.venv\Scripts\ollama-mcp.exe
```

Then allow it without a prompt per call — in `~/.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "mcp__ollama-local__local_edit",
      "mcp__ollama-local__local_explain",
      "mcp__ollama-local__local_verify",
      "mcp__ollama-local__local_status"
    ]
  }
}
```

Without this you get an approval prompt on every delegation and you will stop using it by day three.

## Make Claude actually delegate

This is the step people skip, and it is the one that decides whether any of this pays off. Left alone, Claude will use its own `Edit` tool — it's faster and more certain from where it sits. You have to tell it not to. Add to your project's `CLAUDE.md`:

```markdown
## Local delegation policy

A local model is available via the `ollama-local` MCP server. Route work to it.

DELEGATE to `local_edit` (do not Read the file first):
- docstrings, comments, translations of comments
- type annotations, renames, signature changes you have already decided on
- boilerplate, test scaffolding, applying a pattern across files
- formatting and lint fixes

DELEGATE to `local_explain` instead of Read/Grep when you need a fact, not the file:
- "where is X defined", "what does module Y do", "which call sites pass Z"

DELEGATE to `local_verify` instead of running lint/typecheck through Bash.

KEEP for yourself:
- architecture and design decisions
- debugging, root-cause analysis, anything ambiguous
- cross-file reasoning, security-sensitive code
- any edit where you cannot state the exact change in one paragraph

If a tool returns ESCALATE, the working tree is unchanged — do it yourself.
```

## Configure the gate

Drop a `.ollama-mcp.toml` in each project root. Only `[gate]` really matters; everything else has a working default.

```toml
[gate]
commands = [
  ["ruff", "check", "--quiet", "."],
  ["python", "-m", "pytest", "-x", "-q", "tests/unit"],
]
timeout_s = 180

[models]
fast = "qwen2.5-coder:7b"
deep = "qwen3-coder:30b"
fast_num_ctx = 16384
deep_num_ctx = 32768
keep_alive = "10m"

[limits]
max_iterations = 12
request_timeout_s = 900
max_local_retries = 1

[sandbox]
deny = [".git", ".env", "node_modules", ".venv", ".pem"]

[i18n]
language = "auto"   # "auto" | "en" | "ja"
```

The config file is validated when it loads: an unknown key or a value of the
wrong type stops the server with the file and the key named, rather than
silently falling back to a default three delegations later.

With no config file, the gate falls back to syntax checks on touched files plus an autodetected project check (`ruff`, `tsc --noEmit`, `cargo check`, `go build`).

Keep the gate **fast**. It runs after every delegated edit, and on failure it runs again after one local retry.

## Language

`[i18n] language` (or `OLLAMA_MCP_LANG`) picks the language of the prompt sent to
the local model and of the prose in the receipt. `auto`, the default, decides per
call from the text of the instruction, falling back to the machine locale — so a
task written in Japanese gets the Japanese prompt on an English-locale laptop,
which is the common case.

Status tokens are **never** translated. `APPLIED`, `NOT APPLIED`, `ESCALATE`,
`PASS` and `FAIL` are protocol: every `CLAUDE.md` delegation policy in the wild
says "a tool returning ESCALATE means the working tree is unchanged", and a
localized token would quietly break all of them. Only the sentences around them
change.

Working in Japanese also changes a handful of things you would otherwise have to
discover the hard way — non-UTF-8 consoles, BOMs, decomposed filenames, and what
a token actually costs in CJK. See **[docs/ja/japanese-environment.md](docs/ja/japanese-environment.md)**.

## Tools

| Tool | Use instead of | Returns |
|---|---|---|
| `local_edit` | Read + Edit | files changed, `+n/-m`, gate verdict |
| `local_explain` | Read + Grep | a dense answer with `path:line` cites |
| `local_verify` | Bash lint/typecheck | PASS, or a triaged failure list |
| `local_status` | — | health, models installed, savings so far |

Four tools, deliberately. Every tool schema is re-sent on every request forever; a twelve-tool belt eats back the savings it exists to create.

## Local models that ignore the tool-calling channel

Ollama advertises a `tools` capability for any model whose template can render tool schemas — but plenty of them, `qwen2.5-coder` included, print the call into the message body instead: as bare JSON, in a ```json fence, or wrapped in `<tool_call>` tags. Untreated this looks exactly like *"the model finished without doing anything"*, which is the wrong diagnosis and produces a needless escalation.

So the server parses the message body as a fallback whenever the native field is empty, gated on the tool name being one it actually offers — a model quoting a config file does not get executed. If a model reports DONE without ever calling a write tool, it gets one pointed correction before the task escalates.

## Does it actually save anything?

Every call appends a row to `.ollama-mcp/metrics.jsonl` in the project. `local_status` summarizes it:

```
savings so far: {"calls": 41, "succeeded": 34, "escalated": 7,
                 "estimated_tokens_avoided": 118400,
                 "tokens_spent_on_receipts": 4900,
                 "median_duration_s": 22.4}
```

`estimated_tokens_avoided` is exactly that — an estimate, chars/4 of the file text the local model read that Claude therefore didn't. Treat it as a trend line, not an invoice. The honest check is the escalation rate: **if more than about a third of delegations escalate, you are delegating the wrong class of work.**

## Hardware notes

`qwen3-coder:30b` is MoE (30B total, ~3B active), ~18 GB at Q4_K_M. That matters: unlike a dense 30B, it stays usable when most layers sit in system RAM. `qwen2.5-coder:7b` (4.7 GB) fits entirely in 8 GB VRAM with a 16k context — which is why the fast tier is the default and the deep tier is opt-in per call.

Measured on an RTX 5050 Laptop (8 GB VRAM) + Ryzen 7 260 + 32 GB RAM, adding Google-style docstrings to a 4-function module:

| | fast (`qwen2.5-coder:7b`) | deep (`qwen3-coder:30b`) |
|---|---|---|
| docstring edit | **failed the gate**, rolled back, escalated | **applied**, +37/−0, gate pass |
| time | 39 s | 61 s cold, 26 s warm |
| repo Q&A (`local_explain`) | correct, 3 s | — |

That first row is the design working, not the design failing: a 7B is not reliable at multi-site exact-string edits, the syntax gate caught its broken output, and the working tree was restored byte-for-byte. **Use the fast tier for lookups and single-site edits; reach for `tier="deep"` as soon as an edit touches several places.**

Don't keep both models resident. `keep_alive` pins the last one used; that's 18 GB of RAM for the deep tier.

Note on `num_ctx`: Ollama defaults to 4096 and **truncates silently** past it. This server always sends `num_ctx` explicitly. If you set it higher, remember the KV cache competes with model weights for VRAM.

## Safety

The local model writes to your real working tree. Guards, in order:

1. Every path is resolved and confined to `workspace_root`; traversal is rejected.
2. `[sandbox] deny` blocks `.git`, `.env`, keys, `node_modules`, and anything else you list.
3. Original file contents are snapshotted in memory before the first write and restored automatically if the gate fails.
4. `local_explain` and `local_verify` run with no write tools at all.
5. Edits preserve each file's existing line endings and byte content — no whole-file CRLF churn, and non-ASCII comments (Japanese, accented text) survive a non-UTF-8 console, a BOM, or a legacy Shift-JIS encoding.

Run it in a git repository anyway. In-memory rollback covers gate failures; it does not cover a model that succeeded at the wrong thing.

## Development

```bash
uv pip install -e ".[dev]"
pytest          # 128 tests against a fake Ollama fixture — no GPU, no models needed
ruff check .
```

## License

MIT
