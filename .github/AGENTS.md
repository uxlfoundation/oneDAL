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
