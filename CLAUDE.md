# Local delegation policy

A local model is available through the `ollama-local` MCP server. Route work to it
rather than doing everything yourself — that is the point of this repository.

## Delegate

**`local_edit`** — mechanical, well-specified changes. Do NOT read the files first;
pass the paths inside the instruction and let the local model read them.
- docstrings, comments, translating comments
- type annotations, renames, a signature change you have already decided on
- boilerplate, test scaffolding, applying a pattern across several files
- formatting and lint fixes

**`local_explain`** — instead of Read/Grep, when you need a *fact* rather than the file.
- "where is X defined", "what does module Y do", "which call sites pass Z"

**`local_verify`** — instead of running lint/typecheck/tests through Bash, when you
only need the verdict and what broke.

## Keep for yourself

- architecture and design decisions
- debugging and root-cause analysis
- anything ambiguous, or needing cross-file reasoning
- security-sensitive code
- any edit whose exact shape you cannot state in one paragraph

## Rules

- A tool returning `ESCALATE` means the working tree is unchanged. Do that one yourself.
- Use `tier="deep"` only when the edit needs real code reasoning; it is several times slower.
- If more than about a third of delegations escalate, you are delegating the wrong
  class of work — stop delegating that class.

# Development

```bash
uv pip install -e ".[dev]"
pytest          # 143 tests, fake Ollama fixture, no GPU needed
ruff check .
```

Real end-to-end behaviour can only be verified against a live Ollama on the user's
machine — CI cannot cover it.
