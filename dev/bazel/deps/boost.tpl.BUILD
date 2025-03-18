package(default_visibility = ["//visibility:public"])

cc_library(
    name = "boost",
    srcs = glob([
        "libs/libboost*.a",
    ]),
    hdrs = glob([
        "boost/**/*.h",
        "boost/**/*.hpp",
        "boost/**/*.ipp",
    ]),
    includes = [
        ".",
    ],
    visibility = ["//visibility:public"],
)

