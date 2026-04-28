load("@onedal//dev/bazel:release.bzl",
    "release",
    "release_include",
    "release_extra_file",
)
load("@onedal//dev/bazel:scripts.bzl",
    "generate_vars_sh",
    "generate_pkgconfig",
)

generate_vars_sh(
    name = "release_vars_sh",
    out = "env/vars.sh",
)

generate_pkgconfig(
    name = "release_pkgconfig",
    out = "lib/pkgconfig/onedal.pc",
)

release(
    name = "release",
    include = [
        release_include(
            hdrs = [ "@onedal//cpp/daal:public_includes" ],
            skip_prefix = "cpp/daal/include",
        ),
        release_include(
            hdrs = [ "@onedal//cpp/daal:kernel_defines" ],
            add_prefix = "services/internal",
        ),
        release_include(
            hdrs = [ "@onedal//cpp/oneapi/dal:public_includes" ],
            skip_prefix = "cpp",
        ),
    ],
    lib = [
        "@onedal//cpp/daal:core_static",
        "@onedal//cpp/daal:thread_static",
        "@onedal//cpp/daal:core_dynamic",
        "@onedal//cpp/daal:thread_dynamic",
        "@onedal//cpp/oneapi/dal:static",
        "@onedal//cpp/oneapi/dal:dynamic",
        "@onedal//cpp/oneapi/dal:static_parameters",
        "@onedal//cpp/oneapi/dal:dynamic_parameters",
    ] + select({
        "@config//:release_dpc_enabled": [
            "@onedal//cpp/oneapi/dal:static_dpc",
            "@onedal//cpp/oneapi/dal:dynamic_dpc",
            "@onedal//cpp/oneapi/dal:static_parameters_dpc",
            "@onedal//cpp/oneapi/dal:dynamic_parameters_dpc",
        ],
        "//conditions:default": [],
    }),
    extra_files = [
        release_extra_file(":release_vars_sh",   "env/vars.sh"),
        release_extra_file(":release_pkgconfig", "lib/pkgconfig/onedal.pc"),
    ],
)
