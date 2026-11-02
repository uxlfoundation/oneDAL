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

- [x] **Import libraries, CRLF staging and missing tools without scripts.** The
  Windows toolchain config enables `targets_windows` and
  `supports_interface_shared_libraries`, so `cc_common.link()` declares the
  import library and passes `/IMPLIB:`; the release rule converts line endings
  with `expand_template`; a toolchain configured without `icpx` registers no
  DPC++ tool and `dev/bazel/cc.bzl` reports that during analysis. The MSVC
  runtime is one setting, `--msvc_runtime`, read by the toolchain config as
  well as by the library names and the dependency `select()`s.

### TODO:

- [ ] **Extend compiler support matrix.** Current status:
  |         |        Intel       |        DPC++       |         GCC        |       Clang        |        MSVC        |
  |---------|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
  | Linux   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
  | Windows | :heavy_check_mark: | :heavy_check_mark: |                    |        :x:         | :heavy_check_mark: |

  Intel `icx`/`icpx` is preferred whenever it is on `PATH`; `CC` overrides the
  choice. Linux Clang builds `//:release`, but no CI job selects it, so
  regressions in that flag set are only found by hand. Windows Clang
  (`clang-cl`) is added by #3795.

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

- [ ] **Remove the last helper script.** `toolchains/tools/dpc_link_win.ps1`
  and its `.tpl.bat` launcher keep the Intel driver in DPC++ link actions while
  moving the object list into a response file of their own: the driver flattens
  every argument into one line of the response file it hands to MSVC `link`,
  which rejects lines over 131071 characters (LNK1170). DLL links no longer
  need it, since `dev/bazel/cc/link.bzl` archives their objects into
  `<name>_objects.lib` first; DPC++ executables still pass their objects
  directly. Two script-free options, both needing a Windows DPC++ run to
  validate: archive the objects of executables the same way, or link through
  `lld-link` (`-fuse-ld=lld`), which has no line limit.
