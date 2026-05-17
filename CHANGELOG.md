# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows semantic versioning while it remains in 0.x alpha.

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
