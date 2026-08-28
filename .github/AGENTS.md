# .github Directory: AI Agent Context Guide

> Purpose: rules for AI agents that edit CI and workflow files in `.github/`.

## Do not change `nightly-build.yml`

The `scikit-learn-intelex` repository depends on `nightly-build.yml`. Its CI queries `gh run --workflow "Nightly-build" --status success`. It then downloads these artifacts:

- `__release_lnx`
- `oneDAL_env`
- `__release_win`
- `intel_oneapi_basekit`
- `opencl_rt_installer`

Do not add steps to `nightly-build.yml`. Do not add jobs to `nightly-build.yml`. Do not rename its artifacts. Put all new test steps in `nightly-test.yml`. This includes Bazel steps, comparison steps, CVE scans, and CMake example tests.

If you must rename or remove one of the artifacts above, you must also change `scikit-learn-intelex/.github/workflows/ci.yml` in the same release. Link the `scikit-learn-intelex` repository at https://github.com/uxlfoundation/scikit-learn-intelex.

## Validating a `nightly-test.yml` change without waiting for a nightly

`nightly-test.yml` runs on `workflow_run`, which always executes the copy of the
workflow on `main`. A branch or PR edit to it therefore does not run, and the
jobs it gates on need a finished `Nightly-build` first (Linux ~1h, Windows ~3h).

Do not rebuild the Make release to test a change to the comparison. The Make side
is just an artifact: reuse the `__release_lnx` / `__release_win` (and, for
Windows, `intel_oneapi_basekit`) artifacts of any recent successful
`Nightly-build` via `actions/download-artifact` with an explicit `run-id`, and
pay only for the Bazel release and the comparison step. Two consequences:

- A `push`-triggered run does not inherit the `actions: read` permission that a
  `workflow_run` run gets implicitly. Request it explicitly or the cross-run
  download fails.
- Artifacts live in `uxlfoundation/oneDAL`, so a fork run has to pin
  `repository:` to upstream. It needs no extra secret to read them: upstream is
  public, and `github.token` plus `actions: read` suffices (verified from
  `Alexandr-Solovev/oneDAL` with no secrets configured).

The alternative, documented in the header of `nightly-test.yml` itself, is to
dispatch that workflow with a `source_run_id`; use it when the change must be
exercised through the real job definitions rather than a trimmed copy.

## `.gitignore` swallows workflow files named `bazel-*`

`.gitignore` carries a bare `bazel-*` rule for Bazel's convenience symlinks
(`bazel-bin`, `bazel-out`). It is not anchored to the repository root, so it also
matches, for example, `.github/workflows/bazel-release-parity-check.yml`, and
`git add` reports nothing while the file stays untracked. Name workflow files so
they do not start with `bazel-`.

## Make-vs-Bazel release parity: known toolchain divergences

`dev/release_tests/compare_release_trees.py` compares the two release trees, and
some differences it reports are properties of how the trees are built or
transported rather than build bugs. Before treating a new one as a regression,
check it is not one of these shapes; when it is, extend the script's documented
ignore lists rather than changing the build:

- **Windows exports.** The Make release is built with `vc` (cl) and the Bazel one
  with icx (clang-cl). They disagree about compiler-generated entities: class
  template special members, instantiations naming lambdas, and constructors a
  `dllexport` class inherits with `using Base::Base;` — cl exports all of them,
  clang-cl only those the DLL odr-uses. None can be part of a consumer's link
  surface, because `ONEDAL_EXPORT` expands to `__declspec(dllexport)` only under
  `__ONEDAL_ENABLE_EXPORT__`, which is set solely while oneDAL itself is built.
- **Linux symlinks.** Both trees stage a real `libonedal_core.so.<major>.<minor>`
  plus `.so.<major>` and `.so` aliases (`makefile:975`, `dev/bazel/release.bzl`).
  `actions/upload-artifact` follows symlinks and uploads their contents, so the
  aliases arrive as regular files on whichever side travelled as an artifact.

Out-of-line definitions are the useful signal and stay compared, so a library
that genuinely stopped being linked still fails the comparison.
