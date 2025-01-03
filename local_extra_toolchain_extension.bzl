load("@onedal//dev/bazel/toolchains:extra_toolchain.bzl", "onedal_extra_toolchain")

onedal_extra_toolchain_extension = module_extension(
    implementation = lambda ctx: onedal_extra_toolchain(name = "onedal_extra_toolchain"),
)