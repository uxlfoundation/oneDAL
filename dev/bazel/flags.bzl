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

lnx_cc_common_flags = [
    "-fwrapv",
    "-fstack-protector-strong",
    "-fno-delete-null-pointer-checks",
    "-Werror",
    "-Wno-empty-body",
    "-Wformat",
    "-Wformat-security",
    "-Wreturn-type",
    # TODO: remove in oneDAL 2026.0.
    "-Wno-deprecated-declarations",
]

lnx_cc_pedantic_flags = [
    "-Wall",
    "-Wextra",
    "-pedantic",
    "-Wno-unused-parameter",
    "-Wno-unused-but-set-parameter",
    "-Wno-c2y-extensions",
]

lnx_cc_flags = {
    "common": lnx_cc_common_flags,
    "pedantic": lnx_cc_pedantic_flags,
}

# icx/icpx on Windows use clang-cl spellings; mirrors COMPILER.win.icx in icx.mkl.32e.mk.
# -Qopenmp-simd comes from the runtime_library feature: icx rejects it with the debug CRT.
win_icx_common_flags = [
    "-nologo",
    "-WX",
    # clang-cl disables C++ exceptions by default.
    "-EHsc",
    # Keeps __svml_* references out of the released .lib files (`-Zl.icx` in Make).
    "-Qno-intel-lib",
    "-Wno-deprecated-declarations",
    "-Wno-ignored-attributes",
    "-Wno-empty-body",
    "-Wno-pass-failed",
    # fmt and Catch2 headers discard [[nodiscard]] results.
    "-Wno-unused-result",
    "-Wno-unused-variable",
]

win_icx_pedantic_flags = [
    "-Wall",
    "-Wextra",
    "-Wwritable-strings",
    "-Wno-unused-parameter",
]

win_icx_flags = {
    "common": win_icx_common_flags,
    "pedantic": win_icx_pedantic_flags,
}

# Upstream clang-cl, used for winarm; mirrors COMPILER.win.clang in clang.ref.arm.mk.
# The CRT flag (-MD/-MDd) comes from the toolchain's msvc_runtime_debug feature.
win_clang_common_flags = [
    "-Werror",
    "-Wno-empty-body",
    "-Wreturn-type",
    "-Wno-deprecated-declarations",
    # clang-cl disables C++ exceptions by default.
    "-EHsc",
    # clang-cl reports every vectorization pragma it cannot honor.
    "-Wno-pass-failed",
]

# Make has none for winarm, and under clang-cl `-Wall` means `-Weverything`.
win_clang_pedantic_flags = []

win_clang_flags = {
    "common": win_clang_common_flags,
    "pedantic": win_clang_pedantic_flags,
}

# clang-cl targets the arch of its own binary unless told otherwise.
_WIN_CLANG_TARGET_TRIPLE = {
    "arm": "aarch64-pc-windows-msvc",
    "intel64": "x86_64-pc-windows-msvc",
}

def get_win_clang_target_flags(arch_id):
    return ["--target={}".format(_WIN_CLANG_TARGET_TRIPLE[arch_id])]

def get_default_flags(arch_id, os_id, compiler_id, category = "common"):
    _check_flag_category(category)
    if os_id == "lnx":
        flags = lnx_cc_flags[category]
        if compiler_id == "icx" and category == "common":
            flags = flags + [
                "-qopenmp-simd",
                # Keeps __svml_* references, which gcc cannot resolve, out of static archives.
                "-no-intel-lib",
                "-no-canonical-prefixes",
            ]
        if compiler_id == "icpx":
            flags = flags + [
                "-no-intel-lib",
                "-fsycl",
                "-fno-canonical-system-headers",
                "-no-canonical-prefixes",
                "-Wno-gnu-zero-variadic-macro-arguments",
            ]
        if compiler_id == "icpx" and category == "pedantic":
            flags = flags + ["-Wno-unused-command-line-argument"]
        if compiler_id == "gcc":
            flags = flags + [
                "-Wno-gnu-zero-variadic-macro-arguments",
                "-fopenmp-simd",
            ]
        if compiler_id == "clang":
            flags = flags + [
                # Unhonored vectorization pragmas; icx/icpx honor them, so they keep the warning.
                "-Wno-pass-failed",
            ]
        if compiler_id == "gcc":
            # clang implies it from -fwrapv and reports it as unused.
            flags = flags + ["-fno-strict-overflow"]
        if arch_id in ["arm", "riscv64"]:
            # A single ISA variant, so -march goes on every compile action.
            flags = flags + _get_single_variant_march_flags(arch_id)
            if arch_id == "arm" and compiler_id == "gcc" and category == "common":
                flags = flags + ["-ftree-vectorize"]
        return flags
    if os_id == "win":
        if compiler_id in ["icx", "icpx"]:
            flags = win_icx_flags[category]
            if compiler_id == "icpx" and category == "common":
                flags = flags + ["-fsycl"]
            return flags
        if compiler_id == "clang":
            flags = win_clang_flags[category]
            if category == "common":
                flags = flags + get_win_clang_target_flags(arch_id)
                if arch_id in _SINGLE_VARIANT_MARCH_FLAGS:
                    flags = flags + _get_single_variant_march_flags(arch_id)
            return flags
        # cl uses the rules_cc MSVC auto-config and none of these flags.
        return []
    fail("Unsupported OS")

_SINGLE_VARIANT_MARCH_FLAGS = {
    "arm": ["-march=armv8-a+sve"],
    "riscv64": ["-march=rv64gc_v1p0_zvl128b"],
}

_SINGLE_VARIANT_ISA_ID = {
    "arm": "sve",
    "riscv64": "rv64",
}

def _get_single_variant_march_flags(arch_id):
    return _SINGLE_VARIANT_MARCH_FLAGS[arch_id]

def get_cpu_flags(arch_id, os_id, compiler_id):
    if arch_id in ["arm", "riscv64"]:
        # Repeats the common -march only because flag_group() needs a non-empty list.
        return {
            _SINGLE_VARIANT_ISA_ID[arch_id]: _get_single_variant_march_flags(arch_id),
        }
    sse2 = []
    avx2 = []
    avx512 = []
    if compiler_id == "gcc":
        sse2 = ["-march=nocona"]
        avx2 = ["-march=haswell"]
        avx512 = ["-march=skylake-avx512"]
    elif compiler_id == "clang":
        sse2 = ["-march=nocona"]
        avx2 = ["-march=haswell"]
        avx512 = ["-march=skylake-avx512"]
    elif compiler_id in ["icx", "icpx"]:
        sse2 = ["-march=nocona"]
        avx2 = ["-march=haswell"]
        avx512 = ["-march=skx"]
    return {
        "sse2": sse2,
        "avx2": avx2,
        "avx512": avx512,
    }

def _check_flag_category(category):
    if not category in ["common", "pedantic"]:
        fail("Unsupported compiler flag category '{}' ".format(category) +
             "expected 'common' or 'pedantic'")
