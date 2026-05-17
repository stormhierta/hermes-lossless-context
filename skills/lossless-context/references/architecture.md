# Architecture

Hermes Lossless Context is an external addon. It does not patch Hermes core.

## Modes

- Passive plugin mode: registers tools through Hermes plugin API. No default compression behavior changes.
- ContextEngine mode: user explicitly sets `context.engine: lossless`.

## Store

SQLite file defaults to:

`<HERMES_HOME>/plugins/lossless-context/lossless-context.db`

Tables:

- `conversations`
- `messages`
- `summaries`
- `message_summaries`
- `summary_edges`
- `context_items`
- FTS5 virtual tables when available

## Compression

v0.1 uses deterministic leaf summaries. This is intentional: clean installs do not need API keys beyond the normal Hermes model provider.

Future versions can add an auxiliary LLM summarizer through Hermes's auxiliary client, but this should remain fallback-safe.

## Security boundary

Summary content is injected as user-role reference material with an explicit reference-only prefix. Source messages are retained and expanded through tools.
