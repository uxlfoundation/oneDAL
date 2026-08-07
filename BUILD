load("@onedal//dev/bazel:release.bzl",
    "release",
    "release_include",
    "release_extra_file",
)
load("@onedal//dev/bazel:scripts.bzl",
    "generate_cmake_config",
    "generate_modulefile",
    "generate_pkgconfig",
    "generate_vars_sh",
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

config_setting(
    name = "release_dpc_windows",
    constraint_values = ["@platforms//os:windows"],
    flag_values = {
        "@config//:release_dpc": "True",
    },
)

generate_vars_sh(
    name = "release_vars_sh",
    out = "env/vars.sh",
)

generate_modulefile(
    name = "release_modulefile_dal",
    out = "modulefiles/dal",
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
    srcs = ["@onedal//cpp/oneapi/dal:public_includes"],
)

generate_cmake_config(
    name = "release_cmake_config",
    template = "cmake/templates/oneDALConfig.cmake.in",
    out = "lib/cmake/oneDAL/oneDALConfig.cmake",
)

generate_cmake_config(
    name = "release_cmake_config_version",
    template = "cmake/templates/oneDALConfigVersion.cmake.in",
    out = "lib/cmake/oneDAL/oneDALConfigVersion.cmake",
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
    name = "release_mpi_daal_lst_linux",
    srcs = ["samples/daal/cpp/mpi/daal_lnx.lst"],
)

filegroup(
    name = "release_mpi_makefile_linux",
    srcs = ["samples/daal/cpp/mpi/makefile_lnx"],
)

filegroup(
    name = "release_package_files",
    srcs = glob([
        "examples/cmake/setup_examples.cmake",
        "samples/cmake/*.cmake",
        "samples/daal/cpp/mpi/CMakeLists.txt",
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
    ]) + select({
        ":windows": glob([
            "samples/daal/cpp/mpi/daal.lst.bat",
            "samples/daal/cpp/mpi/launcher.bat",
        ]),
        "//conditions:default": [],
    }),
)

filegroup(
    name = "release_dpc_package_files",
    srcs = glob([
        "samples/oneapi/dpc/ccl/CMakeLists.txt",
        "samples/oneapi/dpc/ccl/sources/*.cpp",
        "samples/oneapi/dpc/ccl/sources/*.hpp",
        "samples/oneapi/dpc/mpi/CMakeLists.txt",
        "samples/oneapi/dpc/mpi/sources/*.cpp",
        "samples/oneapi/dpc/mpi/sources/*.hpp",
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
        "@onedal//cpp/oneapi/dal:static",
        "@onedal//cpp/oneapi/dal:dynamic",
    ] + select({
        ":windows": [],
        "//conditions:default": [
            "@onedal//cpp/oneapi/dal:static_parameters",
            "@onedal//cpp/oneapi/dal:dynamic_parameters",
        ],
    }) + select({
        ":release_dpc_windows": [
            "@onedal//cpp/oneapi/dal:dynamic_dpc",
        ],
        "@config//:release_dpc_enabled": [
            "@onedal//cpp/oneapi/dal:dynamic_dpc",
            "@onedal//cpp/oneapi/dal:dynamic_parameters_dpc",
        ],
        "//conditions:default": [],
    }),
    data = [
        "//data:datasets",
        ":release_package_files",
        "//examples/daal/cpp:release_files",
        "//examples/oneapi/cpp:release_files",
    ] + select({
        "@config//:release_dpc_enabled": [
            ":release_dpc_package_files",
            "//examples/oneapi/dpc:release_files",
        ],
        "//conditions:default": [],
    }),
    extra_files = [
        release_extra_file(":release_vars_sh", "env/vars.sh", windows_dst_path = "env/vars.bat"),
        release_extra_file(":release_pkgconfig", "", windows_dst_path = ""),
        release_extra_file(":release_pkgconfig_dynamic_threading_host", "lib/pkgconfig/dal-dynamic-threading-host.pc"),
        release_extra_file(":release_pkgconfig_static_threading_host", "lib/pkgconfig/dal-static-threading-host.pc"),
        release_extra_file("//deploy/local:config_file", "config/config.txt"),
        release_extra_file(":release_modulefile_dal", "modulefiles/dal", windows_dst_path = ""),
        release_extra_file(":release_mpi_daal_lst_linux", "samples/daal/cpp/mpi/daal.lst", windows_dst_path = ""),
        release_extra_file(":release_mpi_makefile_linux", "samples/daal/cpp/mpi/makefile", windows_dst_path = ""),
        release_extra_file(":release_cmake_config", "lib/cmake/oneDAL/oneDALConfig.cmake"),
        release_extra_file(":release_cmake_config_version", "lib/cmake/oneDAL/oneDALConfigVersion.cmake"),
        release_extra_file(":release_nuspec_devel", "nuspec/inteldal.devel.linux.nuspec", windows_dst_path = "nuspec/inteldal.devel.win-x64.nuspec"),
        release_extra_file(":release_nuspec_redist", "nuspec/inteldal.redist.linux.nuspec", windows_dst_path = "nuspec/inteldal.redist.win-x64.nuspec"),
        release_extra_file(":release_nuspec_static", "nuspec/inteldal.static.linux.nuspec", windows_dst_path = "nuspec/inteldal.static.win-x64.nuspec"),
    ],
)
