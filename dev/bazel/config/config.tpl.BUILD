package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel/config:config.bzl",
    "cpu_info",
    "version_info",
    "config_flag",
    "config_bool_flag",
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
        ":test_link_mode",
        ":test_thread_mode",
    ],
)
