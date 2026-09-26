# Agent Instructions for `.github/instructions`

These files provide Copilot guidance for paths selected by their YAML `applyTo` front matter.

## Rules for Changes

- Keep `applyTo` patterns aligned with every file type the instruction governs, including templates and configuration files.
- Treat overlapping instruction files as additive. Keep broad guidance brief and let scoped files add detail without contradicting it.
- Keep PR review rules limited to source-confirmed correctness, compatibility, ownership, dispatch, error handling, and test coverage. Do not duplicate checks enforced by formatting, license, or other CI jobs.
- Link only to existing files, using paths relative to the instruction file.
- Put durable subsystem context in the nearest `AGENTS.md`; use these files only for Copilot-specific and path-scoped guidance.
