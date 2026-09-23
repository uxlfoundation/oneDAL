# Copilot instructions - oneDAL

Repository guidance lives in `AGENTS.md` files next to the code they govern. Before reviewing or editing a file, read the root `AGENTS.md` and the nearest `AGENTS.md` above the file:

| Path | Guidance |
| --- | --- |
| `cpp/daal/` | `cpp/AGENTS.md`, `cpp/daal/AGENTS.md` |
| `cpp/oneapi/` | `cpp/AGENTS.md`, `cpp/oneapi/AGENTS.md` |
| `dev/bazel/`, `**/BUILD`, `*.bzl`, `MODULE.bazel` | `dev/bazel/AGENTS.md` |
| `makefile`, `dev/make/` | `dev/make/AGENTS.md` |
| `.ci/` | `.ci/AGENTS.md` |
| `.github/` | `.github/AGENTS.md`, `.ci/AGENTS.md` |
| `examples/`, `samples/` | `examples/AGENTS.md` |
| `docs/` | `docs/AGENTS.md` |
| `deploy/` | `deploy/AGENTS.md` |

## Reviewing pull requests
- Check the diff against the "Rules for Changes" sections of the relevant `AGENTS.md` files. Those rules come from recurring maintainer review comments.
- Report at most 10 findings, most severe first. Correctness, CPU dispatch, ABI breaks, and build changes that miss a platform outrank style.
- Don't report what pre-commit or CI already enforces: clang-format, editorconfig, license headers, and the ABI check.
- Only report a finding you have confirmed in the source. Say so when a finding depends on code outside the diff that you couldn't read.
- Don't ask for Doxygen comments on internal code. Do flag documentation that contradicts the implementation.
- Flag PR descriptions that don't match the diff, and PRs that bundle unrelated changes.
- Keep comments short and concrete; include a suggestion block when the fix is small.
