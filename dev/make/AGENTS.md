# AGENTS.md - Make Build Fragments (dev/make/)

## Purpose
Fragments included by the root `makefile`, which drives the production build (see `INSTALL.md`).

## Layout
- `common.mk`: compile, link and packaging commands shared by all platforms
- `deps.mk`, `deps.mkl.mk`, `deps.ref.mk`: third-party dependencies per backend (`BACKEND_CONFIG=mkl|ref`)
- `compiler_definitions/<compiler>[.<backend>].<arch>.mk`: per-compiler flags
- `function_definitions/<plat>.mk`: per-platform helpers (`lnx32e`, `win32e`, `mac32e`, `lnxarm`, `winarm`, `lnxriscv64`)
- `identify_os.sh`: host OS detection

## Rules for Changes
- Compile flags go in `COPT`, link flags in `LOPT`.
- Library binary versions come from `MAJORBINARY` / `MINORBINARY` in `makefile.ver`. The Bazel build mirrors them in `dev/bazel/repos.bzl`; change both together.
- Export and symbol-visibility changes here need the matching change in the Bazel build (`dev/bazel/`), and vice versa.
- Shell and batch script rules are in the root `AGENTS.md`.
