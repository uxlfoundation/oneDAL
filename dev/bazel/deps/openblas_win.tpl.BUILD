package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")

# `.ci/env/openblas.bat` installs through CMake, which nests the headers in
# `include/openblas`, unlike the `make install` layout the Linux script produces.
# Both roots are exposed so either spelling resolves; oneDAL itself declares the
# BLAS/LAPACK prototypes locally and includes none of these.
cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h"], allow_empty = True),
    includes = [
        "include",
        "include/openblas",
    ],
)

# `.ci/env/openblas.bat` builds a static openblas.lib, which Make links too (OPENBLASDIR.libia.win).
cc_library(
    name = "openblas_core",
    srcs = [
            "lib/openblas.lib",
           ],
)

cc_library(
    name = "openblas",
    deps = [
        ":headers",
        ":openblas_core",
    ],
)
