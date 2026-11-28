load("@onedal//dev/bazel/toolchains:rc_toolchain.bzl", "RC_TOOLCHAIN_TYPE", "rc_toolchain")

package(default_visibility = ["//visibility:public"])

rc_toolchain(
    name = "rc_tool",
    files = [%{rc_deps}],
    rc_path = "%{rc_path}",
)

toolchain(
    name = "rc_toolchain_win",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    toolchain = ":rc_tool",
    toolchain_type = RC_TOOLCHAIN_TYPE,
)

toolchain(
    name = "cc_toolchain_win",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    toolchain = "//msvc:cc-compiler-x64_windows",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)
