# Hermes Lossless Context

Opt-in Lossless Context Management addon for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

This package provides a reasonable way for Hermes to understand more of its own context after compaction/compression:

- persistent SQLite message store
- FTS5 search over old session material
- source-preserving leaf summary DAG
- agent-callable recall tools: `lcm_grep`, `lcm_describe`, `lcm_expand`, `lcm_status`
- optional ContextEngine replacement selected explicitly with `context.engine: lossless`

## Safety model

Default Hermes behavior is unchanged unless you enable the plugin.

The addon never deletes source messages during automatic operation. Summaries are reference material only; source messages remain in SQLite and can be expanded through tools.

## Install into a clean Hermes Agent setup

From this repository:

```bash
python -m pip install -e .
python scripts/install_user_plugin.py
```

Then enable the user plugin in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - lossless_context
```

Restart Hermes or start a new session.

## Passive mode

With only the plugin enabled, Hermes gets the `lossless_context` toolset and the recall tools. Because Hermes currently has no passive per-turn observer hook, passive indexing happens when tools are called and when the engine lifecycle sees messages.

Tools:

- `lcm_grep(pattern, mode="full_text", scope="both", limit=20)`
- `lcm_describe(id)`
- `lcm_expand(id, maxMessages=20)`
- `lcm_status()`

## Full ContextEngine mode

To replace the built-in compressor with the lossless engine, add:

```yaml
context:
  engine: lossless
```

This mode is still opt-in. The engine indexes messages, creates deterministic leaf summaries for older turns, and injects those summaries as reference-only context while preserving a recent raw tail.

Environment knobs:

```bash
HERMES_LCM_DB=/path/to/lossless-context.db
HERMES_LCM_THRESHOLD=0.75
HERMES_LCM_PROTECT_FIRST=3
HERMES_LCM_PROTECT_LAST=6
HERMES_LCM_LEAF_CHUNK_TOKENS=20000
HERMES_LCM_MIN_LEAF_MESSAGES=4
```

## Development

```bash
python -m pip install -e '.[dev]'
pytest
python -m build
```

## Limitations in v0.1

- No Hermes core hook is required or used.
- No background/passive observer exists yet, so passive mode cannot index every turn unless Hermes passes messages to a tool or the context engine is active.
- Summaries use a deterministic safe fallback, not an LLM summarizer, to keep clean installs dependency-free.
- No destructive repair, pruning, or transcript rewriting commands are included.

## Security notes

- SQLite queries use parameterized values.
- The only dynamic SQL fragments are fixed internal clauses, not user-controlled table/column names.
- FTS queries are sanitized before MATCH.
- Summary text is wrapped as reference-only context and should not be treated as active instruction.
