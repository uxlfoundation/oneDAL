<!--
******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/-->

### Done:

- [x] **Windows support.** `toolchains/cc_toolchain_win.bzl` configures Intel
  `icx` and, when it is not on `PATH`, MSVC `cl` through rules_cc's
  auto-configuration. `//:release` and `//:release_all` build on Windows in both
  CRT flavours and are covered by CI. Running DPC++ examples and tests there is
  still open, see below.

- [x] **Release to oneAPI structure.** `//:release` writes headers, libraries,
  examples, environment scripts, pkg-config and CMake metadata to
  `bazel-bin/release/daal/latest`. Nightly CI compares that tree against the
  Make one with `dev/release_tests/compare_release_trees.py`.

- [x] **Automatic host architecture identification.** `--cpu=auto`, the default,
  detects the highest instruction set available on the host and adds the `sse2`
  baseline.

### TODO:

- [ ] **Extend compiler support matrix.** Current status:
  |         |        Intel       |        DPC++       |         GCC        |       Clang        |        MSVC        |
  |---------|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
  | Linux   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |        :x:         |                    |
  | Windows | :heavy_check_mark: | :heavy_check_mark: |                    |        :x:         | :heavy_check_mark: |

  Intel `icx`/`icpx` is preferred whenever it is on `PATH`; `CC` overrides the
  choice. `detect_compiler` in `toolchains/common.bzl` recognises `clang`, but
  there is no Clang entry in the flag tables, so that path is untested.

- [ ] **Windows DPC++ execution.** Windows release artifacts can include the
  DPC++ libraries, but the examples and tests that create a SYCL queue are not
  run there yet; that needs device and runtime validation on the CI images.

- [ ] **Toolchain flag tables as data.** The Linux and Windows
  `cc_toolchain_config` rules now share their action groups, their
  attribute-driven features and their rule attributes through
  `toolchains/cc_toolchain_config_common.bzl`. What is left in the two files is
  not duplication: no same-named feature emits the same flags on both
  platforms, so folding them together would put a per-platform branch inside
  every factory. The duplication that is real sits in `toolchains/common.bzl`,
  where the per-compiler, per-OS, per-ISA flag lists are spelled out in
  Starlark control flow and would read better as a table.

- [ ] **Hermetic toolchain.** The compiler, archiver, linker and strip tool are
  absolute paths found with `repo_ctx.which()` at configure time. A build
  therefore depends on the host `PATH`, and nothing invalidates the action cache
  when the compiler behind that path changes. Registering a downloaded toolchain
  would fix both.

- [ ] **Remove the remaining helper scripts.** Four are left, and each waits on
  something Bazel or rules_cc does not expose:
  - `toolchains/tools/dll_to_implib.bat` — `cc_common.link()` does not register
    `supports_interface_shared_libraries` for this toolchain config and cannot
    declare `-IMPLIB:`'s side-effect file as an output, so the import library is
    derived from `dumpbin` output after the link.
  - `toolchains/tools/copy_crlf.bat` — the release tree needs CRLF text files
    and no Bazel action rewrites line endings.
  - `toolchains/tools/dpc_link_win.ps1` and its `.tpl.bat` launcher — keeps the
    Intel driver in DPC++ link actions while spilling long object lists into a
    response file, which the driver needs and Bazel's own param-file support
    does not provide for a wrapped tool.
  - `toolchains/tools/tool_not_found.tpl.sh` and `.tpl.bat` — stand in for an
    optional tool that is not installed, today only the DPC++ compiler, and
    fail the action that reaches them. The failure belongs in analysis instead:
    `@config` already knows whether DPC++ was found, so a DPC++ target could
    `select()` onto an error target the way `daal_module` does for `--stdalloc`
    on non-Linux.
