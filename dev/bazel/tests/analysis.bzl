#===============================================================================
# Copyright contributors to the oneDAL project
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

"""Analysis-time checks of the configured build graph; `bazel test //dev/bazel/tests:all`."""

load("@bazel_skylib//lib:unittest.bzl", "analysistest", "asserts")
load("@onedal//dev/bazel/config:config.bzl", "TARGET_ATTRS", "VERSION_ATTRS", "get_version_info", "target")

_RELEASE_DPC = str(Label("@config//:release_dpc"))
_CPU = str(Label("@config//:cpu"))
_CODE_COVERAGE = str(Label("@config//:code_coverage"))
_BUILD_PARAMETERS_LIB = str(Label("@config//:build_parameters_lib"))

# Every DPC++ graph is analysed on avx2 only; the DPC++ targets stay in the graph without icpx.
_DPC_SETTINGS = {_RELEASE_DPC: True, _CPU: "avx2"}

def _defines(actions):
    """The `-D` arguments of the compile actions, deduplicated."""
    defines = {}
    for action in actions:
        if action.mnemonic != "CppCompile":
            continue
        for arg in action.argv:
            if arg.startswith("-D") or arg.startswith("/D"):
                defines[arg[2:]] = True
    return defines

def _dpc_lp64_test_impl(ctx):
    env = analysistest.begin(ctx)
    defines = _defines(analysistest.target_actions(env))
    asserts.true(env, defines, "no compile action found")
    asserts.true(env, "MKL_LP64" in defines, "the DPC++ library objects compile with MKL_LP64")
    asserts.false(env, "MKL_ILP64" in defines, "the DPC++ library objects are LP64, as Make's")
    return analysistest.end(env)

def _dpc_ilp64_test_impl(ctx):
    env = analysistest.begin(ctx)
    defines = _defines(analysistest.target_actions(env))
    asserts.true(env, "MKL_ILP64" in defines, "the DPC++ MKL test helper keeps MKL_ILP64")
    return analysistest.end(env)

dpc_lp64_test = analysistest.make(_dpc_lp64_test_impl, config_settings = _DPC_SETTINGS)
dpc_ilp64_test = analysistest.make(_dpc_ilp64_test_impl, config_settings = _DPC_SETTINGS)

# --- coverage: Make's -coverage on every icx compile, -Xscoverage on DPC++ links ---

def _coverage_args(env):
    compile_args = []
    link_args = []
    for action in analysistest.target_actions(env):
        if action.mnemonic == "CppCompile":
            compile_args.append(action.argv)
        elif action.mnemonic.startswith("CppLink"):
            link_args.append(action.argv)
    return compile_args, link_args

def _coverage_core_test_impl(ctx):
    env = analysistest.begin(ctx)
    compile_args, _ = _coverage_args(env)
    asserts.true(env, compile_args, "no compile action found")
    for argv in compile_args:
        asserts.true(env, "-coverage" in argv, "every core compile carries -coverage")
        asserts.true(env, "-DGCOV_BUILD" in argv, "every core compile defines GCOV_BUILD")
    return analysistest.end(env)

def _coverage_threading_test_impl(ctx):
    env = analysistest.begin(ctx)
    compile_args, _ = _coverage_args(env)
    asserts.true(env, compile_args, "no compile action found")
    for argv in compile_args:
        asserts.true(env, "-coverage" in argv, "every threading compile carries -coverage")
        asserts.false(env, "-DGCOV_BUILD" in argv, "the threading objects do not define GCOV_BUILD")
    return analysistest.end(env)

def _coverage_dpc_link_test_impl(ctx):
    env = analysistest.begin(ctx)
    _, link_args = _coverage_args(env)
    asserts.true(env, link_args, "no link action found")
    for argv in link_args:
        asserts.true(env, "-Xscoverage" in argv, "the DPC++ link carries -Xscoverage")
    return analysistest.end(env)

coverage_core_test = analysistest.make(
    _coverage_core_test_impl,
    config_settings = {_CODE_COVERAGE: True, _CPU: "avx2"},
)
coverage_threading_test = analysistest.make(
    _coverage_threading_test_impl,
    config_settings = {_CODE_COVERAGE: True, _CPU: "avx2"},
)
coverage_dpc_link_test = analysistest.make(
    _coverage_dpc_link_test_impl,
    config_settings = _DPC_SETTINGS | {_CODE_COVERAGE: True},
)

# --- parameters library layout ---

_DepsInfo = provider(fields = ["direct_lib", "transitive"])

def _deps_aspect_impl(target, ctx):
    transitive = [dep[_DepsInfo].transitive for dep in getattr(ctx.rule.attr, "deps", [])]
    lib = [str(t.label) for t in getattr(ctx.rule.attr, "lib", [])]
    return [_DepsInfo(
        direct_lib = lib,
        transitive = depset([str(target.label)], transitive = transitive),
    )]

# Transitive labels through `deps`, and the `lib` labels of the release rule.
_deps_aspect = aspect(implementation = _deps_aspect_impl, attr_aspects = ["deps"])

_PARAMETER_LIBS = [
    "@@//cpp/oneapi/dal:static_parameters",
    "@@//cpp/oneapi/dal:dynamic_parameters",
    "@@//cpp/oneapi/dal:dynamic_parameters_dpc",
]
_HOST_PARAMETER_MODULES = [
    "@@//cpp/oneapi/dal/algo:parameters",
    "@@//cpp/oneapi/dal/detail/parameters:parameters",
]
_DPC_PARAMETER_MODULES = [
    "@@//cpp/oneapi/dal/algo:parameters_dpc",
    "@@//cpp/oneapi/dal/detail/parameters:parameters_dpc",
]

def _release_parameters_test_impl(ctx):
    env = analysistest.begin(ctx)
    libs = analysistest.target_under_test(env)[_DepsInfo].direct_lib
    for lib in _PARAMETER_LIBS:
        asserts.equals(env, ctx.attr.separate, lib in libs,
                       "{} staged by the release: {}".format(lib, ctx.attr.separate))
    return analysistest.end(env)

def _make_release_parameters_test(value):
    return analysistest.make(
        _release_parameters_test_impl,
        attrs = {"separate": attr.bool(mandatory = True)},
        config_settings = _DPC_SETTINGS | {_BUILD_PARAMETERS_LIB: value},
        extra_target_under_test_aspects = [_deps_aspect],
    )

release_parameters_auto_test = _make_release_parameters_test("auto")
release_parameters_yes_test = _make_release_parameters_test("yes")
release_parameters_no_test = _make_release_parameters_test("no")

def _folded_parameters_test_impl(ctx):
    # Folded layout: the libraries embed their own parameter modules, never the other interface's.
    env = analysistest.begin(ctx)
    deps = analysistest.target_under_test(env)[_DepsInfo].transitive.to_list()
    own = _DPC_PARAMETER_MODULES if ctx.attr.dpc else _HOST_PARAMETER_MODULES
    other = _HOST_PARAMETER_MODULES if ctx.attr.dpc else _DPC_PARAMETER_MODULES
    for module in own:
        asserts.true(env, module in deps, "{} is linked in".format(module))
    for module in other:
        asserts.false(env, module in deps, "{} belongs to the other interface".format(module))
    return analysistest.end(env)

folded_parameters_test = analysistest.make(
    _folded_parameters_test_impl,
    attrs = {"dpc": attr.bool(default = False)},
    config_settings = _DPC_SETTINGS | {_BUILD_PARAMETERS_LIB: "no"},
    extra_target_under_test_aspects = [_deps_aspect],
)

def _windows_parameters_rejected_test_impl(ctx):
    env = analysistest.begin(ctx)
    asserts.expect_failure(env, "--build_parameters_lib=yes is not supported on Windows")
    return analysistest.end(env)

windows_parameters_rejected_test = analysistest.make(
    _windows_parameters_rejected_test_impl,
    expect_failure = True,
    config_settings = {
        _BUILD_PARAMETERS_LIB: "yes",
        "//command_line_option:platforms": str(Label("@config//:windows_analysis_platform")),
    },
)

# --- release tree layout (Linux) ---

_RELEASE_FILES = [
    "include/daal.h",
    "include/services/library_version_info.h",
    "include/services/internal/daal_kernel_defines.h",
    "include/oneapi/dal.hpp",
    "env/vars.sh",
    "modulefiles/dal",
    "config/config.txt",
    "lib/pkgconfig/dal-dynamic-threading-host.pc",
    "lib/pkgconfig/dal-static-threading-host.pc",
    "lib/cmake/oneDAL/oneDALConfig.cmake",
    "lib/cmake/oneDAL/oneDALConfigVersion.cmake",
    "nuspec/inteldal.devel.linux.nuspec",
    "nuspec/inteldal.redist.linux.nuspec",
    "nuspec/inteldal.static.linux.nuspec",
    "samples/daal/cpp/mpi/daal.lst",
    "samples/daal/cpp/mpi/makefile",
]
_RELEASE_LIBS = ["onedal_core", "onedal_thread", "onedal", "onedal_parameters"]
_FOREIGN_HEADER_PREFIXES = ["mkl", "tbb", "dpl_"]

def _release_layout_test_impl(ctx):
    env = analysistest.begin(ctx)
    t = target(ctx)
    vi = get_version_info(ctx)
    root = analysistest.target_under_test(env).label.name + "/"
    paths = {}
    for f in analysistest.target_under_test(env)[DefaultInfo].files.to_list():
        asserts.true(env, f.short_path.startswith(root), "{} is outside the release root".format(f.short_path))
        paths[f.short_path[len(root):]] = True

    expected = ["daal/latest/" + p for p in _RELEASE_FILES]
    lib_dir = "daal/latest/lib/{}/".format(t.arch_dir)
    for lib in _RELEASE_LIBS:
        expected.append(lib_dir + "lib{}.a".format(lib))
        for suffix in [".so", ".so." + vi.binary_major, ".so.{}.{}".format(vi.binary_major, vi.binary_minor)]:
            expected.append(lib_dir + "lib{}{}".format(lib, suffix))
    for path in expected:
        asserts.true(env, path in paths, "{} is staged".format(path))
    asserts.true(env, [p for p in paths if p.startswith("tbb/latest/lib/libtbb")], "TBB ships next to daal/latest")

    for path in paths:
        if not path.startswith("daal/latest/include/"):
            continue
        rel = path[len("daal/latest/include/"):]
        asserts.false(env, rel.startswith("external/") or rel.startswith("+"),
                      "{} comes from an external repository".format(path))
        base = rel.split("/")[-1]
        for prefix in _FOREIGN_HEADER_PREFIXES:
            asserts.false(env, base.startswith(prefix), "{} is a dependency's header".format(path))
    return analysistest.end(env)

release_layout_test = analysistest.make(
    _release_layout_test_impl,
    attrs = TARGET_ATTRS | VERSION_ATTRS,
    config_settings = {_RELEASE_DPC: False, _CPU: "avx2"},
)

def _soname_test_impl(ctx):
    env = analysistest.begin(ctx)
    vi = get_version_info(ctx)
    soname = "-Wl,-soname,lib{}.so.{}".format(ctx.attr.lib_name, vi.binary_major)
    links = [a for a in analysistest.target_actions(env) if a.mnemonic.startswith("CppLink")]
    asserts.true(env, links, "no link action found")
    for action in links:
        asserts.true(env, soname in action.argv, "the link sets {}".format(soname))
    return analysistest.end(env)

soname_test = analysistest.make(
    _soname_test_impl,
    attrs = {
        "lib_name": attr.string(mandatory = True),
    } | VERSION_ATTRS,
    config_settings = {_CPU: "avx2"},
)
