# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows semantic versioning while it remains in 0.x alpha.

## [0.1.2] - 2026-05-17

### Fixed

- Corrected the Hermes package entry-point group from `hermes.plugins` to `hermes_agent.plugins` and pointed it at the plugin module expected by the Hermes loader.
- Fixed `lcm_describe` summary field mapping so `created_at`, `earliest_seq`, and `latest_seq` are reported correctly.
- Hardened `HERMES_LCM_DB` path containment validation.
- Changed injected lossless summary references from `system` to `assistant` messages so historical content is not elevated into active system policy.
- Fixed `lcm_expand.truncated` so exact-limit expansions are not reported as truncated.
- Preserved Unicode search terms in FTS query sanitization.
- Synchronized package `__version__` with project metadata.

### Added

- Regression coverage for summary field mapping, Unicode FTS terms, DB path containment, `lcm_expand` truncation, and non-system summary reference roles.

## [0.1.1] - 2026-05-17

### Fixed

- Added the `hermes.plugins` package entry point so Hermes can discover the Lossless Context plugin after installation.
- Added a `plugin_entry/plugin.yaml` manifest for user-plugin shim installs under `~/.hermes/plugins/lossless_context/`.
- Updated the user-plugin install script to copy/write the manifest, fixing `context.engine: lossless` falling back to the built-in compressor because the plugin was not discovered.

### Changed

- Bumped package and skill metadata from `0.1.0` to `0.1.1`.

### Tests

- Added regression coverage for the user-plugin manifest and package entry point.

## [0.1.0] - 2026-05-17

### Added

- Initial Lossless Context addon package.
- SQLite-backed source message store with FTS search.
- Opt-in `LosslessContextEngine` implementation for Hermes.
- Recall tools: `lcm_grep`, `lcm_describe`, `lcm_expand`, and `lcm_status`.
- User-plugin shim installer and bundled `lossless-context` skill documentation.
