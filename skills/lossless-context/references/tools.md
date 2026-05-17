# Tool Reference

## lcm_grep

Search messages and summaries.

Parameters:

- `pattern` required string
- `mode`: `full_text` or `like`
- `scope`: `messages`, `summaries`, or `both`
- `limit`: 1-100

## lcm_describe

Return summary metadata and source sequence IDs.

Parameters:

- `id`: summary id, e.g. `sum_abc123...`

## lcm_expand

Return source messages linked to a summary.

Parameters:

- `id`: summary id
- `maxMessages`: 1-100

## lcm_status

Return engine status and integrity report.
