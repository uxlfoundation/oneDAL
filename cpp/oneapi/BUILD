load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_global_header_test",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "include_root",
    # All include paths in `*.cpp/*.hpp` files are relative to
    # `cpp` directory (e.g., `oneapi/dal/array.hpp`), so add parent
    # directory to includes search path
    includes = [".."],

    # Compiled part of oneAPI part depends on DAAL sources. Compilation
    # options for the oneAPI part are more strict than for DAAL, it leads
    # to warnings caused by the inclusion DAAL headers in oneAPI sources.
    # To disable warnings from the third-party dependencies compilers recommend
    # to include them via `-isystem` rather than `-I`. `system_includes` forces
    # Bazel to pass the folders bellow via `-isystem` (to learn more, explore
    # includes patching logic in `dev/bazel/cc/compile.bzl`).
    system_includes = [
        "../daal",
        "../daal/include",
    ],
)

dal_module(
    name = "dal_header",
    hdrs = [
        "dal.hpp",
    ],
)

dal_global_header_test(
    name = "dal_hpp_test",
    algo_dir = "dal/algo",
    algo_exclude = [],
    algo_preview = [
        "jaccard",
        "louvain",
        "triangle_counting",
        "shortest_paths",
        "subgraph_isomorphism",
        "connected_components",
    ],
    dal_deps = [
        ":dal_header",
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal:optional",
    ],
)
