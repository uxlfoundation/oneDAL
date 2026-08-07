package(default_visibility = ["//visibility:public"])

# Helper used by dev/bazel/release.bzl to derive a Windows DLL's
# import library (`<name>_dll.lib`) post-link by harvesting the DLL's
# exports via dumpbin and feeding them to `lib /def:`. Mirrors the
# makefile `-IMPLIB:<name>_dll.lib` pattern, which Bazel's
# cc_common.link() cannot expose as a tracked output for us to forward
# into the staged release tree.
exports_files([
    "dll_to_implib.bat",
    "dpc_link_win.ps1",
    "dpc_link_win.tpl.bat",
])
