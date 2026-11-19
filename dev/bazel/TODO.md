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

- [x] **Analysis tests and no build scripts.** `bazel test //dev/bazel/tests:all`
  checks the configured graph with Starlark analysis tests; the build date is
  `--build_date`, so stamping needs no `workspace_status` script; the release
  tree is staged with `expand_template`; `TargetInfo` (`@config//:target`)
  replaces the per-rule constraint attributes.

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
  | Windows | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |

  Intel `icx`/`icpx` is preferred whenever it is on `PATH`; `CC` overrides the
  choice. Linux Clang builds `//:release`, but no CI job selects it, so
  regressions in that flag set are only found by hand. Windows `clang-cl` is
  the ARM64 lane of `ci-win.yml`.

- [ ] **Windows DPC++ execution.** Windows release artifacts can include the
  DPC++ libraries, but the examples and tests that create a SYCL queue are not
  run there yet; that needs device and runtime validation on the CI images.

- [ ] **Toolchain flag tables as data.** The compile action configs, the default
  compile flags and the marker features are shared by
  `toolchains/cc_toolchain_config_common.bzl`; the link configs and every
  flag-emitting feature still differ between Linux and Windows (`-D` vs `/D`,
  `rcsD` vs `/OUT:`). `flags.bzl` spells the per-compiler and per-OS flag lists
  out in control flow; a table per compiler would read better.

- [ ] **Hermetic toolchain.** The compiler, archiver, linker and strip tool are
  absolute paths found with `repo_ctx.which()` at configure time. A build
  therefore depends on the host `PATH`. The toolchain repository writes
  `tool_identity.txt` (the `--version` and, on Linux, the sha256 of each tool) into
  the compile, archive and link inputs and watches the tool binaries, so a
  compiler changed behind the same path re-runs the configuration and misses the
  action cache instead of reusing stale objects. Registering a downloaded
  toolchain would remove the `PATH` dependency too.

- [ ] **Remove the last helper script.** `toolchains/tools/dpc_link_win.ps1`
  and its `.tpl.bat` launcher do two things to the DPC++ link line on Windows:
  they move the objects into a response file of their own, because the Intel
  driver flattens every argument into one line of the response file it hands to
  MSVC `link`, which rejects lines over 131071 characters (LNK1170); and they
  move the `.res` file ahead of `/link`, where the driver takes it as an input.
  DLL links no longer need the first part, since `dev/bazel/cc/link.bzl`
  archives their objects into `<name>_objects.lib`; DPC++ executables still
  pass their objects directly. Script-free options, all needing a Windows DPC++
  run to validate: archive the objects of executables the same way, or link
  through `lld-link` (`-fuse-ld=lld`), which has no line limit. Converting the
  `.res` with `cvtres` into an object of that archive was tried and failed.
  Findings from PR #4042 (Azure 66390, 66418): (1) hypothesis, not isolated:
  the `.res` before `/link` is likely what makes icx honour `/IMPLIB`. Main's
  working link has it there; the run that put a cvtres object after `/link`
  (66390) gave no `<name>.if.lib`, but it also replaced the `.res` with that
  object, so the two changes are not separated. (2) Bazel rejects
  a `.res` as a compilation output (`wrong extension ... objects: o, obj, opb,
  bc`, 66418), so it cannot reach a feature that emits objects before `/link`.
  (3) Untried: a link variable or feature that emits only the `.res` before
  `/link`. `user_link_flags` already names it, after `/link`; `additional_inputs`
  only stages it. The wrapper's comment gives LNK2001 `_DllMainCRTStartup` as
  the reason for the `.res` before `/link`.

- [ ] **No inherited shell environment.** Two actions still run with
  `use_default_shell_env = True`: the Windows `cmd.exe` copy in `release.bzl`
  and the resource compile in `cc.bzl` (`rc.exe`/`llvm-rc` resolved from `PATH`,
  not from the toolchain). Taking the rc tool from the toolchain config, as the
  archiver is via `cc_common.get_tool_for_action`, removes the second. The
  Windows archive merge in `cc/link.bzl` now takes the toolchain's env.
  `dev/bazel/tests/check_starlark_policy.py` (pre-commit hook `starlark-policy`) lists the two
  sites as exceptions and fails on a new one. Still open in the same CI policy
  step: `buildifier` as a required check (the tree is not formatted yet) and a
  stored aquery diff of `//:release` across the release matrix.
