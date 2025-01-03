load("@onedal//dev/bazel/toolchains:cc_toolchain.bzl", "onedal_cc_toolchain")

cc_toolchain_extension = module_extension(
    implementation = lambda ctx: onedal_cc_toolchain(name = "onedal_cc_toolchain"),
)