package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel/config:config.bzl",
    "cpu_info",
    "version_info",
    "config_flag",
    "config_bool_flag",
    "build_parameters_lib_validation",
    "dump_config_info",
    "unsupported_config",
    "target_info",
)

cpu_info(
    name = "cpu",
    auto_cpu = "%{auto_cpu}",
    arch = "%{arch}",
    build_setting_default = "auto",
)

version_info(
    name = "version",
    major = "%{version_major}",
    minor = "%{version_minor}",
    update = "%{version_update}",
    build = "%{version_build}",
    buildrev = "%{version_buildrev}",
    status = "%{version_status}",
    # Binary ABI version for SONAME and shared library symlinks.
    binary_major = "%{version_binary_major}",
    binary_minor = "%{version_binary_minor}",
)

config_flag(
    name = "backend_config",
    build_setting_default = "mkl",
    allowed_build_setting_values = [
        "ref",
        "mkl",
    ],
)

config_setting(
    name = "backend_ref",
    flag_values  = {
        ":backend_config": "ref",
    },
)

# A DLL resolves every symbol at link time, so onedal_core links the ref math backend
# directly (Make: daaldep.math_backend.shared_link_deps).
config_setting(
    name = "backend_ref_windows",
    flag_values = {
        ":backend_config": "ref",
    },
    constraint_values = [
        "@platforms//os:windows",
    ],
)

config_setting(
    name = "backend_config_mkl_linux",
    flag_values = {
        ":backend_config": "mkl",
    },
    constraint_values = [
        "@platforms//os:linux",
    ],
)

# Windows CRT (Make MSVC_RUNTIME_VERSION), independent of --compilation_mode as it renames the
# libraries and their dependencies. The toolchain's msvc_runtime_debug feature follows it.
config_flag(
    name = "msvc_runtime",
    build_setting_default = "release",
    allowed_build_setting_values = [
        "release",
        "debug",
    ],
)

# Make BUILD_PARAMETERS_LIB parity. "auto" means yes on every non-Windows
# target and no on Windows. Explicit yes is rejected on Windows.
config_flag(
    name = "build_parameters_lib",
    build_setting_default = "auto",
    allowed_build_setting_values = [
        "auto",
        "yes",
        "no",
    ],
)

config_setting(
    name = "build_parameters_lib_auto_windows",
    flag_values = {":build_parameters_lib": "auto"},
    constraint_values = ["@platforms//os:windows"],
)

config_setting(
    name = "build_parameters_lib_yes_windows",
    flag_values = {":build_parameters_lib": "yes"},
    constraint_values = ["@platforms//os:windows"],
)

config_setting(
    name = "release_dpc_disabled",
    flag_values = {":release_dpc": "False"},
)

build_parameters_lib_validation(
    name = "validate_build_parameters_lib",
    flag = ":build_parameters_lib",
    target = ":target",
)

target_info(
    name = "target",
    msvc_runtime = ":msvc_runtime",
)

# Make RNG_BACKEND parity; has an effect with backend_config=ref only.
config_flag(
    name = "rng_backend",
    build_setting_default = "ref",
    allowed_build_setting_values = [
        "ref",
        "openrng",
    ],
)

# Gate OpenRNG on this rather than `:rng_backend_openrng`: Make reaches it on the ref backend
# only, and an MKL build would otherwise require OPENRNGROOT.
config_setting(
    name = "rng_backend_openrng_ref",
    flag_values = {
        ":rng_backend": "openrng",
        ":backend_config": "ref",
    },
)

# Cross-compile targets, e.g. `--platforms=@config//:linux_aarch64` with `CC=aarch64-linux-gnu-gcc`.
platform(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
)

platform(
    name = "linux_riscv64",
    constraint_values = [
        "@platforms//cpu:riscv64",
        "@platforms//os:linux",
    ],
)

# Make's `PLAT=winarm`, clang-cl only. Name it to cross-compile from an x86_64 host
# (`CC=clang-cl --target=aarch64-pc-windows-msvc`).
platform(
    name = "windows_arm64",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:windows",
    ],
)

# Toolchain-free target platform used by cross-platform analysis smoke tests.
platform(
    name = "windows_analysis_platform",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
)

config_flag(
    name = "test_link_mode",
    build_setting_default = "dev",
    allowed_build_setting_values = [
        "dev",
        "release_static",
        "release_dynamic",
    ],
)

config_flag(
    name = "device",
    build_setting_default = "auto",
    allowed_build_setting_values = [
        "auto",
        "cpu",
        "gpu",
    ],
)

# Set only by the transition on the deps of `cc_static_lib` on Windows (see
# dev/bazel/cc.bzl). Objects built with it drop the `__DAAL_IMPLEMENTATION` and
# `__ONEDAL_ENABLE_EXPORT__` defines, so their `__declspec(dllexport)` and
# `/EXPORT:` directives do not reach the static archives, matching Make's
# static Windows objects (makefile: `objs_a`, compiled without the DLL defines).
config_bool_flag(
    name = "static_export_free",
    build_setting_default = False,
)

config_setting(
    name = "static_export_free_enabled",
    flag_values  = {
        ":static_export_free": "True",
    },
)

config_bool_flag(
    name = "test_external_datasets",
    build_setting_default = False,
)

config_setting(
    name = "test_external_datasets_enabled",
    flag_values  = {
        ":test_external_datasets": "True",
    },
)

config_bool_flag(
    name = "test_nightly",
    build_setting_default = False,
)

config_setting(
    name = "test_nightly_enabled",
    flag_values  = {
        ":test_nightly": "True",
    },
)

config_bool_flag(
    name = "test_weekly",
    build_setting_default = False,
)

config_setting(
    name = "test_weekly_enabled",
    flag_values  = {
        ":test_weekly": "True",
    },
)

config_bool_flag(
    name = "test_disable_fp64",
    build_setting_default = False,
)

config_setting(
    name = "test_fp64_disabled",
    flag_values  = {
        ":test_disable_fp64": "True",
    },
)

# Builds the DPC++ targets and puts the DPC++ libraries into //:release.
# Defaults to whether icpx is in PATH.
config_bool_flag(
    name = "release_dpc",
    build_setting_default = %{dpc_compiler_found},
)

# No platform has this constraint, so under --release_dpc=false the DPC++ targets that
# require it are skipped, and a direct request names the missing compiler.
constraint_setting(name = "dpc_compiler")

constraint_value(
    name = "dpc_needs_icpx_in_PATH",
    constraint_setting = ":dpc_compiler",
)

config_bool_flag(
    name = "enable_assert",
    build_setting_default = False,
)

config_bool_flag(
    name = "stdalloc",
    build_setting_default = False,
)

config_setting(
    name = "stdalloc_enabled",
    flag_values = {
        ":stdalloc": "True",
    },
    constraint_values = [
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "stdalloc_disabled",
    flag_values = {
        ":stdalloc": "False",
    },
)

unsupported_config(
    name = "stdalloc_non_linux_error",
    message = "--stdalloc=true is supported only when targeting Linux",
)

# Instrument oneDAL sources with the Intel coverage options used by Make.
config_bool_flag(
    name = "code_coverage",
    build_setting_default = False,
)

config_setting(
    name = "assert_enabled",
    flag_values  = {
        ":enable_assert": "True",
    },
)

config_setting(
    name = "release_dpc_enabled",
    flag_values  = {
        ":release_dpc": "True",
    },
)

dump_config_info(
    name = "dump",
    cpu_info = ":cpu",
    version_info = ":version",
    flags = [
        ":build_parameters_lib",
        ":test_link_mode",
        ":code_coverage",
        ":msvc_runtime",
    ],
)
