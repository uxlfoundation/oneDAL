load("@onedal//dev/bazel:release.bzl",
    "release",
    "release_all",
    "release_include",
)
load("@onedal//dev/bazel/config:selects.bzl",
    "parameters_lib_enabled",
    "parameters_lib_separate_value",
    "release_dpc_parameters_lib_enabled",
)
load("@onedal//dev/bazel:scripts.bzl",
    "generate_cmake_config",
    "generate_modulefile",
    "generate_nuspec",
    "generate_pkgconfig",
    "generate_vars_sh",
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
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
    name = "release_pkgconfig_dynamic_threading_host",
    out = "lib/pkgconfig/dal-dynamic-threading-host.pc",
    parameters_lib = parameters_lib_separate_value(),
)

generate_pkgconfig(
    name = "release_pkgconfig_static_threading_host",
    out = "lib/pkgconfig/dal-static-threading-host.pc",
    static = True,
    parameters_lib = parameters_lib_separate_value(),
)

filegroup(
    name = "release_oneapi_includes",
    srcs = ["@onedal//cpp/oneapi/dal:public_includes"],
)

generate_cmake_config(
    name = "release_cmake_config",
    template = "cmake/templates/oneDALConfig.cmake.in",
    out = "lib/cmake/oneDAL/oneDALConfig.cmake",
    parameters_lib = parameters_lib_separate_value(),
)

generate_cmake_config(
    name = "release_cmake_config_version",
    template = "cmake/templates/oneDALConfigVersion.cmake.in",
    out = "lib/cmake/oneDAL/oneDALConfigVersion.cmake",
)

# Expanded like deploy/nuget/prepare_dal_nuget.sh (makefile:1134). The Linux
# or Windows file is chosen by the target platform.
generate_nuspec(
    name = "release_nuspec_devel",
    template = "deploy/nuget/inteldal.nuspec.tpl",
    distr_type = "devel",
)

generate_nuspec(
    name = "release_nuspec_redist",
    template = "deploy/nuget/inteldal.nuspec.tpl",
    distr_type = "redist",
)

generate_nuspec(
    name = "release_nuspec_static",
    template = "deploy/nuget/inteldal.nuspec.tpl",
    distr_type = "static",
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

# Staged on every OS; the per-OS files are added in the select below.
_RELEASE_EXTRA_FILES = {
    "lib/pkgconfig/dal-dynamic-threading-host.pc": ":release_pkgconfig_dynamic_threading_host",
    "lib/pkgconfig/dal-static-threading-host.pc": ":release_pkgconfig_static_threading_host",
    "config/config.txt": "//deploy/local:config_file",
    "lib/cmake/oneDAL/oneDALConfig.cmake": ":release_cmake_config",
    "lib/cmake/oneDAL/oneDALConfigVersion.cmake": ":release_cmake_config_version",
}

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
        # Ships after `public_includes` so the OS-specific header replaces the
        # generic one it shadows (`daal_win.h` -> `include/daal.h`).
        release_include(
            hdrs = [ "@onedal//cpp/daal:os_specific_includes" ],
            skip_prefix = "cpp/daal/include",
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
    ] + parameters_lib_enabled([
        "@onedal//cpp/oneapi/dal:static_parameters",
        "@onedal//cpp/oneapi/dal:dynamic_parameters",
    ]) + select({
        "@config//:release_dpc_enabled": [
            "@onedal//cpp/oneapi/dal:dynamic_dpc",
        ],
        "//conditions:default": [],
    }) + release_dpc_parameters_lib_enabled([
        "@onedal//cpp/oneapi/dal:dynamic_parameters_dpc",
    ]),
    data = [
        "@config//:validate_build_parameters_lib",
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
    extra_files = select({
        ":windows": _RELEASE_EXTRA_FILES | {
            "env/vars.bat": ":release_vars_sh",
            "nuspec/inteldal.devel.win-x64.nuspec": ":release_nuspec_devel",
            "nuspec/inteldal.redist.win-x64.nuspec": ":release_nuspec_redist",
            "nuspec/inteldal.static.win-x64.nuspec": ":release_nuspec_static",
        },
        "//conditions:default": _RELEASE_EXTRA_FILES | {
            "env/vars.sh": ":release_vars_sh",
            "modulefiles/dal": ":release_modulefile_dal",
            "samples/daal/cpp/mpi/daal.lst": ":release_mpi_daal_lst_linux",
            "samples/daal/cpp/mpi/makefile": ":release_mpi_makefile_linux",
            "nuspec/inteldal.devel.linux.nuspec": ":release_nuspec_devel",
            "nuspec/inteldal.redist.linux.nuspec": ":release_nuspec_redist",
            "nuspec/inteldal.static.linux.nuspec": ":release_nuspec_static",
        },
    }),
    # libonedal_thread needs TBB at runtime, so TBB ships with the package as in Make.
    # On Windows the DLLs go under bin/vc_mt and the import libraries under lib/vc_mt.
    dep_runtime = select({
        ":windows": {
            "tbb/latest/bin/vc_mt": "@tbb//:tbb_runtime",
            "tbb/latest/lib/vc_mt": "@tbb//:tbb_import_libs",
        },
        "//conditions:default": {
            "tbb/latest/lib": "@tbb//:tbb_runtime",
        },
    }),
)

# Both Windows CRT flavours in one tree, as two Make runs (MSVC_RUNTIME_VERSION=release
# and =debug) into one RELEASEDIR would give. Elsewhere it equals //:release.
release_all(
    name = "release_all",
    release_target = ":release",
)
