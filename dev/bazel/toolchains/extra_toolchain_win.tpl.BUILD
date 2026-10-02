package(default_visibility = ["//visibility:public"])

load("@onedal//dev/bazel/toolchains:extra_toolchain.bzl", "extra_toolchain")

extra_toolchain(
    name = "extra_tools",
    patch_daal_kernel_defines = "%{patch_daal_kernel_defines}",
)

toolchain(
    name = "extra_tools_win",
    exec_compatible_with = [
        "@platforms//os:windows",
    ],
    # `patch_daal_kernel_defines.cmd` is an arch-independent text patcher run as
    # a build action, not something that produces target-arch code, so it applies
    # to every Windows target arch (x86_64 and ARM64 alike). Mirrors the same
    # reasoning in extra_toolchian_lnx.tpl.BUILD.
    toolchain = ":extra_tools",
    toolchain_type = "@onedal//dev/bazel/toolchains:extra",
)
