# .github Directory - AI Agents Context Guide

> **Purpose**: Rules for AI agents editing CI/workflow files under `.github/`.

## 🛑 Do NOT modify `nightly-build.yml`

`nightly-build.yml` is a load-bearing dependency of the [`scikit-learn-intelex`](https://github.com/uxlfoundation/scikit-learn-intelex) repo, which queries `gh run --workflow "Nightly-build" --status success` and downloads its `__release_lnx`, `oneDAL_env`, `__release_win`, `intel_oneapi_basekit`, and `opencl_rt_installer` artifacts.

Do NOT add steps, jobs, or artifact renames to `nightly-build.yml`. All new test steps (Bazel, comparisons, CVE scans, CMake example tests, etc.) belong in `nightly-test.yml`. Removing or renaming any of the listed artifacts requires a coordinated change in `scikit-learn-intelex/.github/workflows/ci.yml`.
