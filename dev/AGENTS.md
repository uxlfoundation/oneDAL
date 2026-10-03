# AGENTS.md - Development Tools and Build Systems (dev/)

## Purpose
Build systems and developer tooling. Make (root `makefile`) produces releases; Bazel builds and runs tests; CMake only builds the examples against an installed release.

## Layout
- `bazel/`: Bazel macros and dependency rules; see `dev/bazel/AGENTS.md`
- `make/`: fragments included by the root `makefile`; see `dev/make/AGENTS.md`
- `release_tests/`: checks run against a built release tree (`compare_release_trees.py`, package metadata)
- `l0_tools/`: Level Zero GPU utilities
- `docker/`: development container (`onedal-dev.Dockerfile`)

## Rules for Changes
- A build change usually has to land in both Make and Bazel. Library versions, exports and symbol visibility must match between them.
- BUILD files use the `dal_module`, `dal_test_suite` and `daal_module` macros, never bare `cc_library` / `cc_test`.
- The Bazel version is pinned in `.bazelversion`; don't hardcode it elsewhere.
- Toolchain and dependency setup (oneAPI compilers, oneMKL, oneTBB) is in `INSTALL.md`; don't restate it here.

## Make

```bash
make -f makefile daal oneapi_c PLAT=lnx32e -j$(nproc)
```

- Targets: `daal`, `daal_c`, `oneapi` (`oneapi_c` + `oneapi_dpc`), `onedal`, `onedal_c`, `onedal_dpc`. `make -f makefile help` lists all targets and variables.
- `PLAT`: `lnx32e`, `win32e`, `mac32e`, `lnxarm`, `winarm`, `lnxriscv64`.
- `COMPILER`: `icx` (x86-64 default), `gnu`, `clang`, `vc`; the allowed set per platform is in `make/function_definitions/`.
- `REQCPU`: subset of `sse2 avx2 avx512` on x86-64, `sve` on ARM, `rv64` on RISC-V.
- `BACKEND_CONFIG`: `mkl` (default on x86-64) or `ref` (OpenBLAS; default on ARM and RISC-V).

## Bazel

```bash
bazel test --config=host //cpp/oneapi/dal/algo/pca:tests      # one algorithm, CPU only
bazel test --config=dpc --device=gpu //cpp/oneapi/dal:tests   # DPC++ on GPU
```

See `dev/bazel/README.md` and `dev/bazel/AGENTS.md`.

## CMake

There is no root `CMakeLists.txt`. Build the examples against a release tree:

```bash
source __release_lnx/daal/latest/env/vars.sh
cd examples/oneapi/cpp
cmake -B build -S . -DONEDAL_LINK=dynamic
cmake --build build --parallel
```
