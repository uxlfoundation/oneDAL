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

# `.ci/env/openblas.bat` builds OpenBLAS with `-DBUILD_SHARED_LIBS=OFF`, so
# `lib/openblas.lib` is a static archive and its symbols end up inside oneDAL's
# own binaries. Make links exactly the same file on Windows
# (`releaseopen_blas.LIBS_A := $(OPENBLASDIR)/lib/openblas.lib` through
# `OPENBLASDIR.libia.win` in dev/make/deps.ref.mk).
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

# OpenBLAS is linked statically here too, so nothing has to travel with the
# binaries at run time. The target exists so `_test_runtime_data()` in
# dev/bazel/dal.bzl can name it unconditionally, same as in
# `openblas.tpl.BUILD`.
filegroup(
    name = "openblas_runtime",
    srcs = [],
)
