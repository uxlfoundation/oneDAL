package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel/config:config.bzl",
    "cpu_info",
    "version_info",
    "config_flag",
    "config_bool_flag",
    "dump_config_info",
)

cpu_info(
    name = "cpu",
    auto_cpu = "%{auto_cpu}",
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
    build_setting_default = False,
)

config_bool_flag(
    name = "enable_assert",
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
        ":test_link_mode",
        ":test_thread_mode",
    ],
)
