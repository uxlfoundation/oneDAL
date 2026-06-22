package(default_visibility = ["//visibility:public"])

# Helper used by dev/bazel/release.bzl to derive a Windows DLL's
# import library (`<name>_dll.lib`) post-link by harvesting the DLL's
# exports via dumpbin and feeding them to `lib /def:`. Mirrors the
# makefile `-IMPLIB:<name>_dll.lib` pattern, which Bazel's
# cc_common.link() cannot expose as a tracked output for us to forward
# into the staged release tree.
exports_files(["dll_to_implib.bat"])

# Windows analogue of merge_static_libs_lnx.tpl.sh — thin wrapper around
# the discovered `lib`/`llvm-lib` archiver, staged into the toolchain
# repo by `_create_ar_merge_tool` in cc_toolchain_win.bzl with the
# archiver path baked in. Wired through the `cpp_merge_static_libraries`
# action so dev/bazel/cc/link.bzl uses the same code path on Windows
# and Linux.
exports_files(["merge_static_libs_win.tpl.bat"])
