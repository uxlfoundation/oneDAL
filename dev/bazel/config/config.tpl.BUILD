package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel/config:config.bzl",
    "cpu_info",
    "version_info",
    "config_flag",
    "config_bool_flag",
    "build_parameters_lib_validation",
    "dump_config_info",
    "unsupported_config",
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

config_setting(
    name = "backend_config_mkl_linux",
    flag_values = {
        ":backend_config": "mkl",
    },
    constraint_values = [
        "@platforms//os:linux",
    ],
)

# Windows MSVC runtime library selection, equivalent to the Makefile's
# `MSVC_RUNTIME_VERSION`. Deliberately independent of `--compilation_mode`
# (Make `REQDBG`): the CRT flavour changes the library file names and the
# `msvcrt`/`tbb` variants that get linked, while REQDBG only adds debug
# info and assertions. Ignored on non-Windows platforms.
#
# Keep in sync with the `msvc_runtime_debug` toolchain feature — use
# `--config=mdd` (see .bazelrc), which sets both.
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
)

# Make RNG_BACKEND parity (dev/make/deps.ref.mk RNG_OPENRNG). Only meaningful
# together with backend_config=ref; selecting "openrng" under backend_config=mkl
# has no effect since the ref RNG shim it swaps out is unused in that case.
# `config_flag` above auto-generates the matching `:rng_backend_openrng` /
# `:rng_backend_ref` config_setting targets.
config_flag(
    name = "rng_backend",
    build_setting_default = "ref",
    allowed_build_setting_values = [
        "ref",
        "openrng",
    ],
)

# Use this, not the bare `:rng_backend_openrng`, to gate the OpenRNG define and
# dependency: RNG_OPENRNG lives in dev/make/deps.ref.mk, so Make can only reach
# it on the ref backend, and .ci/scripts/build.sh guards `--use-openrng yes`
# behind `backend_config == ref` too. Without the backend_config term,
# `--rng_backend=openrng --backend_config=mkl` would require OPENRNGROOT and
# link libopenrng into an MKL build whose ref RNG shim is never compiled.
config_setting(
    name = "rng_backend_openrng_ref",
    flag_values = {
        ":rng_backend": "openrng",
        ":backend_config": "ref",
    },
)

# Target platforms for cross-compiling to non-x86 Linux. Pass e.g.
# `--platforms=@config//:linux_aarch64` together with
# `CC=aarch64-linux-gnu-gcc` (or riscv64-linux-gnu-gcc) so the exec platform
# (the build machine) stays x86_64 while the target platform switches arch;
# this is what selects the cross cc_toolchain registered in
# cc_toolchain_lnx.tpl.BUILD.
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

# Windows on ARM64 target platform, matching Make's `PLAT=winarm`. The
# toolchain registered for it in cc_toolchain_win.tpl.BUILD is clang-cl based
# (icx has no AArch64 target), so pass this together with clang-cl on PATH.
# On an ARM64 Windows host it is also the auto-detected default; naming it
# explicitly is what makes an x86_64-host cross-compile
# (`CC=clang-cl --target=aarch64-pc-windows-msvc`) resolve.
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
    name = "test_thread_mode",
    build_setting_default = "par",
    allowed_build_setting_values = [
        "par",
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

config_bool_flag(
    name = "release_dpc",
    build_setting_default = True,
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
        ":test_thread_mode",
        ":msvc_runtime",
    ],
)
