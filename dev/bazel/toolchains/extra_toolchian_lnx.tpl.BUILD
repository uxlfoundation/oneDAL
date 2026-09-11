package(default_visibility = ["//visibility:public"])

load("@onedal//dev/bazel/toolchains:extra_toolchain.bzl", "extra_toolchain")

extra_toolchain(
    name = "extra_tools",
    patch_daal_kernel_defines = "%{patch_daal_kernel_defines}",
)

toolchain(
    name = "extra_tools_lnx",
    exec_compatible_with = [
        "@platforms//os:linux",
    ],
    # `patch_daal_kernel_defines.sh` is an arch-independent text patcher run
    # as a build action, not something that produces target-arch code, so it
    # applies regardless of target platform (including cross-compiles to
    # aarch64/riscv64 where the target platform is not x86_64).
    toolchain = ":extra_tools",
    toolchain_type = "@onedal//dev/bazel/toolchains:extra",
)
