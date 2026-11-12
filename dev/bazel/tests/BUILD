load("@onedal//dev/bazel/tests:analysis.bzl",
    "coverage_core_test",
    "coverage_dpc_link_test",
    "coverage_threading_test",
    "dpc_ilp64_test",
    "dpc_lp64_test",
    "folded_parameters_test",
    "release_layout_test",
    "release_parameters_auto_test",
    "release_parameters_no_test",
    "release_parameters_yes_test",
    "soname_test",
    "windows_parameters_rejected_test",
)

package(default_visibility = ["//visibility:public"])

exports_files([
    "isa_coverage_test.sh",
    "mkl_linkage_test.sh",
])

# The DPC++ library objects are LP64 as Make's; the DPC++ MKL test helper keeps ILP64.
dpc_lp64_test(
    name = "dpc_lp64_test",
    target_under_test = "@onedal//cpp/oneapi/dal/backend/primitives/blas:blas_dpc",
)

dpc_ilp64_test(
    name = "dpc_ilp64_test",
    target_under_test = "@onedal//cpp/oneapi/dal/test/engine/mkl:mkl_dpc",
)

# --build_parameters_lib: auto and yes stage the parameter libraries, no folds them in.
release_parameters_auto_test(
    name = "release_parameters_auto_test",
    target_under_test = "@onedal//:release",
    separate = True,
)

release_parameters_yes_test(
    name = "release_parameters_yes_test",
    target_under_test = "@onedal//:release",
    separate = True,
)

release_parameters_no_test(
    name = "release_parameters_no_test",
    target_under_test = "@onedal//:release",
    separate = False,
)

[folded_parameters_test(
    name = "folded_parameters_{}_test".format(name),
    target_under_test = "@onedal//cpp/oneapi/dal:" + name,
    dpc = name.endswith("_dpc"),
) for name in ["static", "dynamic", "static_dpc", "dynamic_dpc"]]

windows_parameters_rejected_test(
    name = "windows_parameters_rejected_test",
    target_under_test = "@config//:validate_build_parameters_lib",
)

# The release tree holds the versioned libraries and their symlinks, the metadata files and no
# dependency headers; TBB ships next to it.
release_layout_test(
    name = "release_layout_test",
    target_under_test = "@onedal//:release",
    target_compatible_with = ["@platforms//os:linux"],
)

[soname_test(
    name = "soname_{}_test".format(name),
    target_under_test = "@onedal//cpp/daal:" + name,
    lib_name = lib_name,
    target_compatible_with = ["@platforms//os:linux"],
) for name, lib_name in [("core_dynamic", "onedal_core"), ("thread_dynamic", "onedal_thread")]]

# Coverage needs the icx toolchain, so only the coverage CI step runs these (manual).
coverage_core_test(
    name = "coverage_core_test",
    target_under_test = "@onedal//cpp/daal:services",
    tags = ["manual"],
)

coverage_threading_test(
    name = "coverage_threading_test",
    target_under_test = "@onedal//cpp/daal:threading_tbb",
    tags = ["manual"],
)

coverage_dpc_link_test(
    name = "coverage_dpc_link_test",
    target_under_test = "@onedal//cpp/oneapi/dal:dynamic_dpc",
    tags = ["manual"],
)

# The DPC++ link check is not in the suite: CI runs it on manually queued pipelines only.
test_suite(
    name = "coverage_tests",
    tags = ["manual"],
    tests = [
        ":coverage_core_test",
        ":coverage_threading_test",
    ],
)
