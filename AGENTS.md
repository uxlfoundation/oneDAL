# AGENTS.md - oneDAL

oneDAL (oneAPI Data Analytics Library) is a C++ machine learning library with two interfaces: DAAL (`cpp/daal`, CPU) and oneAPI (`cpp/oneapi`, CPU and SYCL GPU). It is built with Make (release builds) and Bazel (tests), and is the backend for [scikit-learn-intelex](https://github.com/uxlfoundation/scikit-learn-intelex).

## Repository Structure

```
oneDAL/
├── cpp/            # Library sources
│   ├── daal/       # DAAL interface and CPU kernels
│   └── oneapi/     # oneAPI interface (C++ and DPC++)
├── dev/            # Build tooling: Bazel rules (dev/bazel), Make fragments (dev/make)
├── examples/       # Examples for the DAAL and oneAPI interfaces
├── samples/        # Distributed (MPI/CCL) samples
├── docs/           # Documentation sources
├── data/           # Datasets used by examples and tests
├── deploy/         # Packaging and environment scripts
├── cmake/          # CMake config templates for installed releases
├── conda-recipe/   # Conda package recipe
└── .ci/            # CI pipelines, environment setup and build/test scripts
```

## Directory Guides

Read the `AGENTS.md` nearest the files you change:

- [cpp/AGENTS.md](cpp/AGENTS.md): C++ implementation details and patterns
- [cpp/daal/AGENTS.md](cpp/daal/AGENTS.md): Traditional DAAL interface context
- [cpp/oneapi/AGENTS.md](cpp/oneapi/AGENTS.md): Modern oneAPI interface context
- [dev/AGENTS.md](dev/AGENTS.md): Development tools and build system context
- [dev/bazel/AGENTS.md](dev/bazel/AGENTS.md): Bazel build system specifics
- [dev/make/AGENTS.md](dev/make/AGENTS.md): Make build fragments
- [docs/AGENTS.md](docs/AGENTS.md): Documentation structure and guidelines
- [examples/AGENTS.md](examples/AGENTS.md): Example code patterns and usage
- [deploy/AGENTS.md](deploy/AGENTS.md): Deployment and distribution context
- [.ci/AGENTS.md](.ci/AGENTS.md): CI/CD infrastructure context
- [.github/AGENTS.md](.github/AGENTS.md): Workflow constraints (`nightly-build.yml`)
- [.github/instructions/AGENTS.md](.github/instructions/AGENTS.md): Copilot instruction scopes and maintenance

## Conventions
- C++17; no C++20/23 features.
- clang-format configs are per source tree (`cpp/daal/`, `cpp/oneapi/`, `examples/*/`, `samples/*/`, `dev/l0_tools/`); there is no root `.clang-format`.
- Parallelize through the oneDAL threading layer, never TBB directly.
- Optimized kernels dispatch on CPU features; see `docs/source/contribution/cpu_features.rst`.

## Rules for Changes

These come from recurring maintainer review comments. Directory-specific rules are in the nearest `AGENTS.md`.

- Comments describe the code as it will be once merged. Don't reference discarded approaches, narrate the change, or mention "this PR".
- Keep comments short and plain: explain why in one line when one line is enough, and in two sentences rather than a paragraph. Agent-written comments have historically been bloated and hard to read; don't restate the code, hedge, or add emphasis.
- One PR, one logical change. Drive-by fixes, renames and mechanical changes (formatting, generated code, mass renames) go in their own PRs.
- Search before adding a helper, constant table or validation routine. Extend the existing one and name it in the PR description.
- Don't add a lock, critical section, guard or redundant check unless you can name the failure it prevents.
- A bug fix comes with a test that fails without the fix.
- Don't hardcode versions, URLs or paths that have a source of truth (`makefile.ver`, `MODULE.bazel`, `.github/renovate.json`).
- New files use the header `Copyright contributors to the oneDAL project`. Leave existing headers alone.
- ASCII only in source, comments and docs. Keep each file's existing line endings.
- Scripts with a `#!/bin/sh` shebang use POSIX `sh` only. `.bat` files follow `cmd.exe` quoting; don't mix PowerShell and CMD syntax.
- Bash scripts start with `set -euo pipefail`, use `mkdir -p`, and don't silence failures with a bare `|| true`.

## Verification Before You Push

### Format and style (blocking: Azure `FormatterChecks`)

```bash
pip install pre-commit && pre-commit install   # one-time
pre-commit run --all-files
editorconfig-checker
```

The pre-commit hook checks the same files as CI, with the same clang-format version (20.1.8; other versions format differently). To run exactly what CI runs, without modifying files: `CLANG_FORMAT_EXE=clang-format-20 .ci/scripts/clang-format.sh`.

### Tests (Bazel)

```bash
bazel test --config=host //cpp/oneapi/dal/algo/<algo>:tests   # one algorithm, CPU only
bazel test --config=host //cpp/oneapi/dal:tests               # oneAPI interface, CPU only
bazel test --config=dpc --device=gpu //cpp/oneapi/dal:tests   # DPC++ on GPU
```

Without `--config`, Bazel builds and runs all tests, including DPC++ ones that need the Intel DPC++ compiler. See `dev/bazel/README.md`.

### Full build (Make)

```bash
make -f makefile daal oneapi_c PLAT=lnx32e -j$(nproc)
```

See `INSTALL.md` for other platforms and build variants.

### Where the checks live

| Check | System | Config |
| --- | --- | --- |
| clang-format, editorconfig-checker | Azure DevOps | `.ci/pipeline/ci.yml` (`FormatterChecks`) |
| Make (GNU/MKL, LLVM/OpenBLAS rv64, VC, Intel), Bazel, release compare, sklearnex | Azure DevOps | `.ci/pipeline/ci.yml` |
| Make + DPC++ (icx), ABI check, Make GNU/MKL conda | GitHub Actions | `.github/workflows/ci.yml` |
| Windows (incl. arm64) | GitHub Actions | `.github/workflows/ci-win.yml` |
| aarch64 | GitHub Actions | `.github/workflows/ci-aarch64.yml` |
| License headers | GitHub Actions | `.github/workflows/skywalking-eyes.yml` |
| Bazel Linux/Windows | GitHub Actions (nightly) | `.github/workflows/nightly-test.yml` |

Style is not gated in GitHub Actions: a green Actions run does not mean formatting passes.

## Further Reading
- `CONTRIBUTING.md`, `INSTALL.md`
- `docs/source/contribution/coding_guide.rst`, `docs/source/contribution/threading.rst`
