load("@onedal//dev/bazel:release.bzl",
    "release",
    "release_include",
    "release_extra_file",
)
load("@onedal//dev/bazel:scripts.bzl",
    "generate_vars_sh",
    "generate_pkgconfig",
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

generate_vars_sh(
    name = "release_vars_sh",
    out = "env/vars.sh",
)

generate_pkgconfig(
    name = "release_pkgconfig",
    out = "lib/pkgconfig/onedal.pc",
)

generate_pkgconfig(
    name = "release_pkgconfig_dynamic_threading_host",
    out = "lib/pkgconfig/dal-dynamic-threading-host.pc",
)

generate_pkgconfig(
    name = "release_pkgconfig_static_threading_host",
    out = "lib/pkgconfig/dal-static-threading-host.pc",
    static = True,
)

filegroup(
    name = "release_oneapi_includes",
    srcs = select({
        ":windows": [],
        "//conditions:default": ["@onedal//cpp/oneapi/dal:public_includes"],
    }),
)

filegroup(
    name = "release_cmake_config",
    srcs = ["cmake/templates/oneDALConfig.cmake.in"],
)

filegroup(
    name = "release_cmake_config_version",
    srcs = ["cmake/templates/oneDALConfigVersion.cmake.in"],
)

filegroup(
    name = "release_nuspec_devel",
    srcs = ["deploy/nuget/inteldal.nuspec.tpl"],
)

filegroup(
    name = "release_nuspec_redist",
    srcs = ["deploy/nuget/inteldal.nuspec.tpl"],
)

filegroup(
    name = "release_nuspec_static",
    srcs = ["deploy/nuget/inteldal.nuspec.tpl"],
)

filegroup(
    name = "release_package_files",
    srcs = glob([
        "examples/cmake/setup_examples.cmake",
        "samples/cmake/*.cmake",
        "samples/daal/cpp/mpi/CMakeLists.txt",
        "samples/daal/cpp/mpi/daal.lst.bat",
        "samples/daal/cpp/mpi/launcher.bat",
        "samples/daal/cpp/mpi/license.txt",
        "samples/daal/cpp/mpi/resources/*",
        "samples/daal/cpp/mpi/sources/*.cpp",
        "samples/daal/cpp/mpi/sources/*.h",
        "samples/oneapi/cpp/ccl/CMakeLists.txt",
        "samples/oneapi/cpp/ccl/sources/*.cpp",
        "samples/oneapi/cpp/ccl/sources/*.hpp",
        "samples/oneapi/cpp/mpi/CMakeLists.txt",
        "samples/oneapi/cpp/mpi/sources/*.cpp",
        "samples/oneapi/cpp/mpi/sources/*.hpp",
    ]),
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
            hdrs = [ ":release_oneapi_includes" ],
            skip_prefix = "cpp",
        ),
    ],
    lib = [
        "@onedal//cpp/daal:core_static",
        "@onedal//cpp/daal:thread_static",
        "@onedal//cpp/daal:core_dynamic",
        "@onedal//cpp/daal:thread_dynamic",
    ] + select({
        ":windows": [],
        "//conditions:default": [
            "@onedal//cpp/oneapi/dal:static",
            "@onedal//cpp/oneapi/dal:dynamic",
            "@onedal//cpp/oneapi/dal:static_parameters",
            "@onedal//cpp/oneapi/dal:dynamic_parameters",
        ],
    }) + select({
        "@config//:release_dpc_enabled": [
            "@onedal//cpp/oneapi/dal:static_dpc",
            "@onedal//cpp/oneapi/dal:dynamic_dpc",
            "@onedal//cpp/oneapi/dal:static_parameters_dpc",
            "@onedal//cpp/oneapi/dal:dynamic_parameters_dpc",
        ],
        "//conditions:default": [],
    }),
    data = [
        "//data:datasets",
        ":release_package_files",
        "//examples/daal/cpp:release_files",
    ],
    extra_files = [
        release_extra_file(":release_vars_sh", "env/vars.sh", windows_dst_path = "env/vars.bat"),
        release_extra_file(":release_pkgconfig", "lib/pkgconfig/onedal.pc", windows_dst_path = ""),
        release_extra_file(":release_pkgconfig_dynamic_threading_host", "", windows_dst_path = "lib/pkgconfig/dal-dynamic-threading-host.pc"),
        release_extra_file(":release_pkgconfig_static_threading_host", "", windows_dst_path = "lib/pkgconfig/dal-static-threading-host.pc"),
        release_extra_file("//deploy/local:config_file", "config/config.txt"),
        release_extra_file(":release_cmake_config", "lib/cmake/oneDAL/oneDALConfig.cmake"),
        release_extra_file(":release_cmake_config_version", "lib/cmake/oneDAL/oneDALConfigVersion.cmake"),
        release_extra_file(":release_nuspec_devel", "nuspec/inteldal.devel.win-x64.nuspec", windows_dst_path = "nuspec/inteldal.devel.win-x64.nuspec"),
        release_extra_file(":release_nuspec_redist", "nuspec/inteldal.redist.win-x64.nuspec", windows_dst_path = "nuspec/inteldal.redist.win-x64.nuspec"),
        release_extra_file(":release_nuspec_static", "nuspec/inteldal.static.win-x64.nuspec", windows_dst_path = "nuspec/inteldal.static.win-x64.nuspec"),
    ],
)
