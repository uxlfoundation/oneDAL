package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "headers",
    # oneMath ships `.hpp` headers plus the `.hxx` files they include; it ships
    # no `.h` at all, so that pattern has to tolerate matching nothing rather
    # than fail the glob.
    hdrs = glob(
        [
            "include/**/*.h",
            "include/**/*.hpp",
            "include/**/*.hxx",
        ],
        allow_empty = True,
    ),
    includes = [
        "include",
    ],
)

cc_library(
    name = "onemath_dpc",
    # Only the run-time dispatching library is linked; the dispatcher loads the
    # per-domain backend libraries itself (see onemath.bzl).
    srcs = glob([
        "lib/libonemath.so*",
    ]),
    linkopts = [
        # Same cap as `@mkl//:mkl_dpc`: a fixed 16 gives the best trade-off
        # between device-code link speedup and memory used, and is clamped to
        # `nproc` on smaller machines.
        "-fsycl-max-parallel-link-jobs=16",
    ],
    deps = [
        ":headers",
        "@opencl//:opencl_binary",
    ],
    # Switches `dal/backend/math_backend.hpp` from <oneapi/mkl.hpp> to
    # <oneapi/math.hpp>. Carried by the dependency rather than by the algorithm
    # targets so the define cannot drift from the library being linked.
    defines = [
        "ONEDAL_MATH_BACKEND_ONEMATH",
    ],
)
