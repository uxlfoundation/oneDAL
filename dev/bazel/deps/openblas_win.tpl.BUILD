package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h"], allow_empty = True),
    includes = [ "include" ],
)

# `.ci/env/openblas.bat` builds OpenBLAS with `-DBUILD_SHARED_LIBS=ON`, so what
# lands in `lib/` is the import library for `bin/openblas.dll`, not a static
# archive. Make links exactly the same file on Windows
# (`releaseopen_blas.LIBS_A := $(OPENBLASDIR)/lib/openblas.lib` through
# `OPENBLASDIR.libia.win` in dev/make/deps.ref.mk), so the DLL has to travel
# with anything that links it.
cc_library(
    name = "openblas_core",
    srcs = [
            "lib/openblas.lib",
           ],
    data = [
            "bin/openblas.dll",
           ],
)

cc_library(
    name = "openblas",
    deps = [
        ":headers",
        ":openblas_core",
    ],
)
