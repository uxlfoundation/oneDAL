#===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

load("@onedal//dev/bazel:cc.bzl",
    "cc_module",
    "cc_static_lib",
    "cc_dynamic_lib",
    "cc_test",
    "ModuleInfo",
)
load("@onedal//dev/bazel/deps:mpi.bzl",
    "mpi_test",
)

load("@onedal//dev/bazel/deps:ccl.bzl",
    "ccl_test",
)
load("@onedal//dev/bazel:release.bzl",
    "headers_filter",
)
load("@onedal//dev/bazel:utils.bzl",
    "sets",
    "paths",
    "utils",
)
load("@onedal//dev/bazel/config:config.bzl",
    "CpuInfo",
)

def dal_module(name, hdrs=[], srcs=[], dal_deps=[], extra_deps=[],
               host_hdrs=[], host_srcs=[], host_deps=[], dpc_hdrs=[],
               dpc_srcs=[], dpc_deps=[], auto=False, auto_exclude=[],
               compile_as=[ "c++", "dpc++" ], **kwargs):
    # TODO: Check `compile_as` parameter
    if auto:
        hpp_filt = ["**/*.hpp"] + auto_exclude
        cpp_filt = ["**/*.cpp"] + auto_exclude
        dpc_filt = ["**/*_dpc.cpp"] + auto_exclude
        test_filt = ["**/*_test*", "**/test/**"] + auto_exclude
        hdrs_all = native.glob(hpp_filt, exclude=test_filt)
        dpc_auto_hdrs = hdrs_all
        dpc_auto_srcs = native.glob(cpp_filt, exclude=test_filt)
        host_auto_hdrs = hdrs_all
        host_auto_srcs = native.glob(cpp_filt, exclude=test_filt + dpc_filt)
    else:
        host_auto_hdrs = []
        host_auto_srcs = []
        dpc_auto_hdrs = []
        dpc_auto_srcs = []
    if "c++" in compile_as:
        _dal_module(
            name = name,
            hdrs = hdrs + host_auto_hdrs + host_hdrs,
            srcs = srcs + host_auto_srcs + host_srcs,
            deps = dal_deps + host_deps + extra_deps,
            **kwargs,
        )
    if "dpc++" in compile_as:
        _dal_module(
            name = name + "_dpc",
            hdrs = hdrs + dpc_auto_hdrs + dpc_hdrs,
            srcs = srcs + dpc_auto_srcs + dpc_srcs,
            deps = _get_dpc_deps(dal_deps) + dpc_deps + extra_deps,
            is_dpc = True,
            **kwargs,
        )

def dal_test_module(name, dal_deps=[], dal_test_deps=[], **kwargs):
    dal_module(
        name = name,
        dal_deps = _test_link_mode_deps(dal_deps) + dal_test_deps,
        testonly = True,
        **kwargs,
    )

def dal_collect_modules(name, root, modules, dal_deps=[], **kwargs):
    module_deps = []
    for module_path in modules:
        module_name = module_path.replace("/", "_")
        module_label = "{0}/{1}".format(root, module_path)
        dal_module(
            name = module_name,
            hdrs = native.glob(["{}*.hpp".format(module_path)]),
            dal_deps = [ module_label ],
        )
        module_deps.append(":" + module_name)
    dal_module(
        name = name,
        dal_deps = dal_deps + module_deps,
        **kwargs,
    )

def dal_public_includes(name, dal_deps=[], **kwargs):
    headers_filter(
        name = name,
        deps = dal_deps + _get_dpc_deps(dal_deps),
        include = [
            "oneapi/",
        ],
        exclude = [
            "backend/",
            "test/",
            "bazel-",
        ],
    )

def dal_static_lib(name, lib_name, dal_deps=[], host_deps=[],
                   dpc_deps=[], extra_deps=[], lib_tags=["dal"],
                   features=[], **kwargs):
    cc_static_lib(
        name = name,
        lib_name = lib_name,
        lib_tags = lib_tags,
        deps = dal_deps + extra_deps + host_deps,
        **kwargs
    )
    cc_static_lib(
        name = name + "_dpc",
        features = features + [ "dpc++" ],
        lib_name = lib_name + "_dpc",
        lib_tags = lib_tags,
        deps = _get_dpc_deps(dal_deps) + extra_deps + dpc_deps,
        **kwargs
    )

def dal_dynamic_lib(name, lib_name, dal_deps=[], host_deps=[],
                    dpc_deps=[], extra_deps=[], lib_tags=["dal"],
                    features=[], **kwargs):
    cc_dynamic_lib(
        name = name,
        lib_name = lib_name,
        lib_tags = lib_tags,
        deps = dal_deps + extra_deps + host_deps,
        **kwargs
    )
    cc_dynamic_lib(
        name = name + "_dpc",
        features = features + [ "dpc++" ],
        lib_name = lib_name + "_dpc",
        lib_tags = lib_tags,
        deps = _get_dpc_deps(dal_deps) + extra_deps + dpc_deps,
        **kwargs
    )

def dal_test(name, hdrs=[], srcs=[], dal_deps=[], dal_test_deps=[],
             extra_deps=[], host_hdrs=[], host_srcs=[], host_deps=[],
             dpc_hdrs=[], dpc_srcs=[], dpc_deps=[], compile_as=[ "c++", "dpc++" ],
             framework="catch2", data=[], tags=[], private=False,
             mpi=False, ccl=False, mpi_ranks=0, args=[], **kwargs):
    # TODO: Check `compile_as` parameter
    # TODO: Refactor this rule once decision on the tests structure is made
    if not framework in ["catch2", "none"]:
        fail("Unknown test framework '{}' in test rule '{}'".format(framework, name))
    is_catch2 = framework == "catch2"
    module_name = "_" + name
    dal_module(
        name = module_name,
        hdrs = hdrs,
        srcs = srcs,
        host_hdrs = host_hdrs,
        host_srcs = host_srcs,
        host_deps = host_deps,
        dpc_hdrs = dpc_hdrs,
        dpc_srcs = dpc_srcs,
        dpc_deps = dpc_deps,
        compile_as = compile_as,
        dal_deps = (
            dal_test_deps +
            _test_link_mode_deps(dal_deps)
        ) + ([
            "@onedal//cpp/oneapi/dal/test/engine:common",
            "@onedal//cpp/oneapi/dal/test/engine:catch2_main",
        ] if is_catch2 else []) + ([
            "@onedal//cpp/oneapi/dal/test/engine:mpi",
        ] if mpi else []) + ([
            "@onedal//cpp/oneapi/dal/test/engine:ccl",
        ] if ccl else []),
        extra_deps = _test_deps_on_daal() + extra_deps,
        testonly = True,
        **kwargs,
    )
    iface_access_tag = "private" if private else "public"
    test_args = _expand_select(
        _test_eternal_datasets_args(framework) +
        _test_filter_args(framework) +
        _test_device_args() +
        args
    )
    # Tests need to be marked as manual to prevent inclusion of all tests to
    # `empty_test` suites, more detail:
    # https://docs.bazel.build/versions/4.1.0/be/general.html#test_suite_args
    common_tags = [ "manual" ]

    tests_for_test_suite = []
    if "c++" in compile_as:
        _dal_cc_test(
            name = name + "_host",
            mpi = mpi,
            ccl = ccl,
            mpi_ranks = mpi_ranks,
            deps = [ ":" + module_name ],
            data = data,
            tags = common_tags + tags + ["host", iface_access_tag],
            args = test_args,
        )
        tests_for_test_suite.append(name + "_host")
    if "dpc++" in compile_as:
        _dal_cc_test(
            name = name + "_dpc",
            features = [ "dpc++" ],
            mpi = mpi,
            ccl = ccl,
            mpi_ranks = mpi_ranks,
            deps = [
                ":" + module_name + "_dpc",
                # TODO: Remove once all GPU algorithms are migrated to DPC++
                "@opencl//:opencl_binary",
            ],
            data = data,
            tags = common_tags + tags + ["dpc", iface_access_tag],
            args = test_args,
        )
        tests_for_test_suite.append(name + "_dpc")
    native.test_suite(
        name = name,
        tests = tests_for_test_suite,
    )

def dal_test_suite(name, srcs=[], tests=[],
                   compile_as=[ "c++", "dpc++" ], **kwargs):
    targets = []
    for test_file in srcs:
        target = test_file.replace(".cpp", "").replace("/", "_")
        if target.endswith("_dpc"):
            is_dpc_only = ("dpc++" in compile_as) and (not "c++" in compile_as)
            if is_dpc_only:
                # We need to remove `_dpc` suffix here as `dal_test` rule
                # adds `_dpc` suffix if `compile_as` attribute contains 'dpc++' target.
                # Otherwise the generated test will have '_dpc_dpc' suffix.
                target = target.replace("_dpc", "")
            else:
                utils.warn("Test name ends with '_dpc' suffix but compiled for both " +
                           "C++ and DPC++. Please check 'compile_as' attribute of the " +
                           "'dal_test_suite(name = {})'. ".format(name))
        dal_test(
            name = target,
            srcs = [test_file],
            compile_as = compile_as,
            **kwargs,
        )
        targets.append(":" + target)
    native.test_suite(
        name = name,
        tests = tests + targets,
    )

def dal_collect_test_suites(name, root, modules=[], target="tests", tests=[], **kwargs):
    test_deps = []
    for module_name in modules:
        test_label = "{0}/{1}:{2}".format(root, module_name, target)
        test_deps.append(test_label)
    dal_test_suite(
        name = name,
        tests = tests + test_deps,
        **kwargs,
    )

def dal_collect_parameters(name, root, modules=[], target="parameters", dal_deps=[], **kwargs):
    module_deps = []
    for module_name in modules:
        module_label = "{0}/{1}:{2}".format(root, module_name, target)
        module_deps.append(module_label)
    dal_module(
        name = name,
        dal_deps = dal_deps + module_deps,
        **kwargs,
    )

def dal_example(name, dal_deps=[], **kwargs):
    dal_test(
        name = name,
        dal_deps = [
            "@onedal//cpp/oneapi/dal:core",
            "@onedal//cpp/oneapi/dal/io",
        ] + dal_deps,
        framework = "none",
        **kwargs,
    )

def dal_example_suite(name, srcs, **kwargs):
    suite_deps = []
    for src in srcs:
        _, alg_name, src_file = src.rsplit('/', 2)
        example_name, _ = paths.split_extension(src_file)
        dal_example(
            name = example_name,
            srcs = [ src ],
            **kwargs,
        )
        suite_deps.append(":" + example_name)
    native.test_suite(
        name = name,
        tests = suite_deps,
    )

def dal_algo_example_suite(algos, dal_deps=[], **kwargs):
    for algo in algos:
        dal_example_suite(
            name = algo,
            srcs = native.glob(["source/{}/*.cpp".format(algo)]),
            dal_deps = dal_deps + [
                "@onedal//cpp/oneapi/dal/algo:{}".format(algo),
            ],
            **kwargs,
        )

def _test_link_mode_deps(dal_deps):
    return _select({
        "@config//:test_link_mode_dev": dal_deps,
        "@config//:test_link_mode_release_static": [
            "@onedal_release//:onedal_static",
        ],
        "@config//:test_link_mode_release_dynamic": [
            "@onedal_release//:onedal_dynamic",
        ],
    })

def _test_deps_on_daal():
    return _select({
        "@config//:test_link_mode_dev": [
            "@onedal//cpp/daal:threading_static",
        ],
        "@config//:test_link_mode_release_static": [
            "@onedal_release//:core_static",
            "@onedal_release//:parameters_static",
            "@onedal//cpp/daal:threading_release_static",
        ],
        "@config//:test_link_mode_release_dynamic": [
            "@onedal_release//:core_dynamic",
            "@onedal_release//:parameters_dynamic",
            "@onedal//cpp/daal:threading_release_dynamic",
        ],
    })

def _test_device_args():
    return _select({
        "@config//:device_cpu": [
            "--device=cpu",
        ],
        "@config//:device_gpu": [
            "--device=gpu",
        ],
        "//conditions:default": [],
    })

def _test_eternal_datasets_args(framework):
    if framework == "catch2":
        return _select({
            "@config//:test_external_datasets_enabled": [],
            "//conditions:default": [
                "~[external-dataset]",
            ],
        })
    return []

def _test_filter_args(framework):
    if framework == "catch2":
        return _select({
            "@config//:test_nightly_enabled": ["~[weekly]"],
            "@config//:test_weekly_enabled": [],
            "//conditions:default": [
                "~[nightly]", "~[weekly]",
            ],
        })
    return []

def _dal_generate_cpu_dispatcher_impl(ctx):
    cpus = sets.make(ctx.attr._cpus[CpuInfo].enabled)
    content = (
        "// DO NOT EDIT: file is auto-generated on build time\n" +
        "// DO NOT PUT THIS FILE TO SVC: file is auto-generated on build time\n" +
        "// CPU detection logic specified in dev/bazel/config.bzl file\n" +
        "\n" +
        ("#define ONEDAL_CPU_DISPATCH_SSE42\n"      if sets.contains(cpus, "sse42")      else "") +
        ("#define ONEDAL_CPU_DISPATCH_AVX2\n"       if sets.contains(cpus, "avx2")       else "") +
        ("#define ONEDAL_CPU_DISPATCH_AVX512\n"     if sets.contains(cpus, "avx512")     else "")
    )
    kernel_defines = ctx.actions.declare_file(ctx.attr.out)
    ctx.actions.write(kernel_defines, content)
    return [ DefaultInfo(files=depset([ kernel_defines ])) ]

dal_generate_cpu_dispatcher = rule(
    implementation = _dal_generate_cpu_dispatcher_impl,
    output_to_genfiles = True,
    attrs = {
        "out": attr.string(mandatory=True),
        "_cpus": attr.label(
            default = "@config//:cpu",
        ),
    },
)

def dal_global_header_test(name, algo_dir, algo_exclude=[], algo_preview=[], dal_deps=[]):
    _generate_global_header_test_cpp(
        name = "_global_header_test_" + name,
        algo_dir = algo_dir,
        algo_exclude = algo_exclude,
        algo_preview = algo_preview,
        deps = dal_deps,
    )
    dal_test(
        name = name,
        srcs = [
            ":_global_header_test_" + name,
        ],
        dal_deps = dal_deps,
    )

def _collect_algorithm_names(algo_dir, deps):
    algo_names = []
    for dep in deps:
        if not ModuleInfo in dep:
            continue
        headers = dep[ModuleInfo].compilation_context.headers.to_list()
        for header in headers:
            header_path = header.path
            header_name = paths.basename(header_path)
            if (algo_dir + "/" + header_name) in header_path:
                algo_name, _ = paths.split_extension(header_name)
                algo_names.append(algo_name)
    return algo_names

def _generate_global_header_test_cpp_impl(ctx):
    algo_dir = paths.normalize(ctx.attr.algo_dir)
    algo_names = _collect_algorithm_names(algo_dir, ctx.attr.deps)
    filtered_algo_names = sets.to_list(
        sets.difference(sets.make(algo_names),
                        sets.make(ctx.attr.algo_exclude))
    )
    preview_algo_names = sets.make(ctx.attr.algo_preview)
    test_file = ctx.actions.declare_file("{}.cpp".format(ctx.label.name))
    content = "#include \"oneapi/dal.hpp\"\n"
    # Add comments for people who may open this file to understand root cause of an error
    content += (
        "\n" +
        "// This is automatically generated file for testing `dal.hpp`.\n" +
        "// If you encountered error compiling this file, it means not all the algorithms\n" +
        "// declared in `{}` are included to `dal.hpp`. Make sure all algorithm are included\n".format(algo_dir) +
        "// or add exception to `algo_exclude` list in `{}`.\n".format(ctx.build_file_path) +
        "\n"
    )
    for algo_name in filtered_algo_names:
        if sets.contains(preview_algo_names, algo_name):
            content += "using namespace oneapi::dal::preview::{};\n".format(algo_name)
        else:
            content += "using namespace oneapi::dal::{};\n".format(algo_name)
    # Linkers may complain about empty object files, so
    # add dummy function. It's more safe to declare static function, but
    # compiler warns about unused functions in this case
    content += "\nvoid __headers_test_dummy__() {}\n"
    ctx.actions.write(test_file, content)
    return [ DefaultInfo(files=depset([ test_file ])) ]

_generate_global_header_test_cpp = rule(
    implementation = _generate_global_header_test_cpp_impl,
    output_to_genfiles = True,
    attrs = {
        "algo_dir": attr.string(mandatory=True),
        "algo_exclude": attr.string_list(),
        "algo_preview": attr.string_list(),
        "deps": attr.label_list(mandatory=True),
    },
)

def _dal_module(name, lib_tag="dal", is_dpc=False, features=[],
                local_defines=[], deps=[], **kwargs):
    cc_module(
        name = name,
        lib_tag = lib_tag,
        features = [ "pedantic", "c++17" ] + features + (
            ["dpc++"] if is_dpc else []
        ),
        cpu_defines = {
            "sse2":   [ "__CPU_TAG__=__CPU_TAG_SSE2__"   ],
            "sse42":  [ "__CPU_TAG__=__CPU_TAG_SSE42__"  ],
            "avx2":   [ "__CPU_TAG__=__CPU_TAG_AVX2__"   ],
            "avx512": [ "__CPU_TAG__=__CPU_TAG_AVX512__" ],
        },
        local_defines = local_defines + ([
            "ONEDAL_DATA_PARALLEL"
        ] if is_dpc else []) + select({
            "@config//:test_fp64_disabled": [
                "ONEDAL_DISABLE_FP64_TESTS=1",
            ],
            "//conditions:default": [],
        }) + select({
            "@config//:assert_enabled": [
                "ONEDAL_ENABLE_ASSERT=1",
            ],
            "//conditions:default": [],
        }) + select({
            "@config//:backend_ref": [
                "DAAL_REF",
                "ONEDAL_REF",
            ],
            "//conditions:default": [],
        }),
        deps = _expand_select(deps),
        **kwargs,
    )

def _dal_cc_test(name, mpi=False, ccl = False, mpi_ranks=0, **kwargs):
    if mpi:
        if mpi_ranks <= 0:
            fail("Test is marked as MPI, you must provide `mpi_ranks` " +
                 "attribute with the valid number of MPI ranks ")
        cc_test(
            name = "_mpi_" + name,
            **kwargs,
        )
        mpi_test(
            name = name,
            src = "_mpi_" + name,
            mpi_ranks = mpi_ranks,
            mpiexec = "@mpi//:mpiexec",
            fi = "@mpi//:fi",
        )
    elif ccl:
        if mpi_ranks <= 0:
            fail("Test is marked as CCL, you must provide `mpi_ranks` " +
                 "attribute with the valid number of MPI ranks ")
        cc_test(
            name = "_ccl_" + name,
            **kwargs,
        )
        ccl_test(
            name = name,
            src = "_ccl_" + name,
            mpi_ranks = mpi_ranks,
            mpiexec = "@mpi//:mpiexec",
            fi = "@mpi//:fi",
        )
    else:
        cc_test(
            name = name,
            **kwargs,
        )

def _select(x):
    return [x]

def _normalize_dep(dep):
    if dep.rfind(":") > 0:
        return dep
    last_slash_index = dep.rfind("/")
    if last_slash_index < 0:
        return dep
    package_name = dep[last_slash_index + 1:].rstrip()
    return dep + ":" + package_name

def _get_dpc_dep_name(name):
    return _normalize_dep(name) + "_dpc"

def _get_dpc_deps(deps):
    normalized = []
    for dep in deps:
        if type(dep) == "dict":
            normalized_dep = {}
            for key, value in dep.items():
                normalized_dep[key] = [
                    _get_dpc_dep_name(x) for x in value
                ]
            normalized.append(normalized_dep)
        else:
            normalized.append(_get_dpc_dep_name(dep))
    return normalized

def _expand_select(deps):
    expanded = []
    for dep in deps:
        if type(dep) == 'dict':
            expanded += select(dep)
        else:
            expanded += [dep]
    return expanded
