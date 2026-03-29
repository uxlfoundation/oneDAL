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

# This BUILD.bazel file is based on the original from the Catch2 project.
# Catch2 is licensed under the Boost Software License 1.0 (BSL-1.0).
#
# Original source: https://github.com/catchorg/Catch2
# License text: see LICENSE.MIT in this repository
#
# Modifications for oneDAL:
#   - Adjusted substitutions specific for oneDAL (e.g. enabled CATCH_CONFIG_NO_POSIX_SIGNALS)
#   - Added comments related to enabling/disabling macros
#   - Removed `license()` rule and `package()` directive to avoid dependency on `rules_license`
#
# SPDX-License-Identifier: BSL-1.0

load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@rules_cc//cc:defs.bzl", "cc_library")

# oneDAL version of catch2 bazel build file
expand_template(
    name = "catch_user_config",
    out = "catch2/catch_user_config.hpp",
    substitutions = {
        "@CATCH_CONFIG_CONSOLE_WIDTH@": "80",
        "@CATCH_CONFIG_DEFAULT_REPORTER@": "console",
        "#cmakedefine CATCH_CONFIG_ANDROID_LOGWRITE": "",
        "#cmakedefine CATCH_CONFIG_BAZEL_SUPPORT": "#define CATCH_CONFIG_BAZEL_SUPPORT",
        "#cmakedefine CATCH_CONFIG_COLOUR_WIN32": "",
        "#cmakedefine CATCH_CONFIG_COUNTER": "",
        "#cmakedefine CATCH_CONFIG_CPP11_TO_STRING": "",
        "#cmakedefine CATCH_CONFIG_CPP17_BYTE": "",
        "#cmakedefine CATCH_CONFIG_CPP17_OPTIONAL": "",
        "#cmakedefine CATCH_CONFIG_CPP17_STRING_VIEW": "",
        "#cmakedefine CATCH_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS": "",
        "#cmakedefine CATCH_CONFIG_CPP17_VARIANT": "",
        "#cmakedefine CATCH_CONFIG_DEPRECATION_ANNOTATIONS": "",
        "#cmakedefine CATCH_CONFIG_DISABLE_EXCEPTIONS_CUSTOM_HANDLER": "",
# Disables unexpected exceptions handing in Catch2.
# It is easier to debug exception via GDB if there is handler.
        "#cmakedefine CATCH_CONFIG_DISABLE_EXCEPTIONS": "#define CATCH_CONFIG_DISABLE_EXCEPTIONS",
        "#cmakedefine CATCH_CONFIG_DISABLE_STRINGIFICATION": "",
        "#cmakedefine CATCH_CONFIG_DISABLE": "",
        "#cmakedefine CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS": "",
        "#cmakedefine CATCH_CONFIG_ENABLE_OPTIONAL_STRINGMAKER": "",
        "#cmakedefine CATCH_CONFIG_ENABLE_PAIR_STRINGMAKER": "",
        "#cmakedefine CATCH_CONFIG_ENABLE_TUPLE_STRINGMAKER": "",
        "#cmakedefine CATCH_CONFIG_ENABLE_VARIANT_STRINGMAKER": "",
        "#cmakedefine CATCH_CONFIG_EXPERIMENTAL_REDIRECT": "",
        "#cmakedefine CATCH_CONFIG_FALLBACK_STRINGIFIER @CATCH_CONFIG_FALLBACK_STRINGIFIER@": "",
        "#cmakedefine CATCH_CONFIG_FAST_COMPILE": "",
        "#cmakedefine CATCH_CONFIG_GETENV": "",
        "#cmakedefine CATCH_CONFIG_GLOBAL_NEXTAFTER": "",
        "#cmakedefine CATCH_CONFIG_NO_ANDROID_LOGWRITE": "",
        "#cmakedefine CATCH_CONFIG_NO_COLOUR_WIN32": "",
        "#cmakedefine CATCH_CONFIG_NO_COUNTER": "",
        "#cmakedefine CATCH_CONFIG_NO_CPP11_TO_STRING": "",
        "#cmakedefine CATCH_CONFIG_NO_CPP17_BYTE": "",
        "#cmakedefine CATCH_CONFIG_NO_CPP17_OPTIONAL": "",
        "#cmakedefine CATCH_CONFIG_NO_CPP17_STRING_VIEW": "",
        "#cmakedefine CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS": "",
        "#cmakedefine CATCH_CONFIG_NO_CPP17_VARIANT": "",
        "#cmakedefine CATCH_CONFIG_NO_DEPRECATION_ANNOTATIONS": "",
        "#cmakedefine CATCH_CONFIG_NO_GETENV": "",
        "#cmakedefine CATCH_CONFIG_NO_GLOBAL_NEXTAFTER": "",
# CATCH_CONFIG_POSIX_SIGNAL enables handling of POSIX signals.
# For unknown reason user-defined handlers for signals
# (see https://en.wikipedia.org/wiki/C_signal_handling)
# catches SIGSEGV signal when USM pointer is accessed on host.
# To make USM work, we disable signal handling in Catch2.
        "#cmakedefine CATCH_CONFIG_NO_POSIX_SIGNALS": "#define CATCH_CONFIG_NO_POSIX_SIGNALS",
        "#cmakedefine CATCH_CONFIG_NO_USE_ASYNC": "",
        "#cmakedefine CATCH_CONFIG_NO_EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT": "",
        "#cmakedefine CATCH_CONFIG_NO_WCHAR": "",
        "#cmakedefine CATCH_CONFIG_NO_WINDOWS_SEH": "",
        "#cmakedefine CATCH_CONFIG_NOSTDOUT": "",
        "#cmakedefine CATCH_CONFIG_POSIX_SIGNALS": "",
        "#cmakedefine CATCH_CONFIG_PREFIX_ALL": "",
        "#cmakedefine CATCH_CONFIG_PREFIX_MESSAGES": "",
        "#cmakedefine CATCH_CONFIG_SHARED_LIBRARY": "",
        "#cmakedefine CATCH_CONFIG_EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT": "",
        "#cmakedefine CATCH_CONFIG_USE_ASYNC": "",
        "#cmakedefine CATCH_CONFIG_WCHAR": "",
        "#cmakedefine CATCH_CONFIG_WINDOWS_CRTDBG": "",
        "#cmakedefine CATCH_CONFIG_WINDOWS_SEH": "",
        "#cmakedefine CATCH_CONFIG_USE_BUILTIN_CONSTANT_P": "",
        "#cmakedefine CATCH_CONFIG_NO_USE_BUILTIN_CONSTANT_P": "",
        "#cmakedefine CATCH_CONFIG_EXPERIMENTAL_THREAD_SAFE_ASSERTIONS": "",
        "#cmakedefine CATCH_CONFIG_NO_EXPERIMENTAL_THREAD_SAFE_ASSERTIONS": "",
    },
    template = "src/catch2/catch_user_config.hpp.in",
)

# Generated header library, modifies the include prefix to account for
# generation path so that we can include <catch2/catch_user_config.hpp>
# correctly.
cc_library(
    name = "catch2_generated",
    hdrs = ["catch2/catch_user_config.hpp"],
    include_prefix = ".",  # to manipulate -I of dependencies
    visibility = ["//visibility:public"],
)

# Static library, without main.
cc_library(
    name = "catch2",
    srcs = glob(
        ["src/catch2/**/*.cpp"],
        exclude = ["src/catch2/internal/catch_main.cpp"],
    ),
    hdrs = glob(["src/catch2/**/*.hpp"]),
    includes = ["src/"],
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [":catch2_generated"],
)

# Static library, with main.
cc_library(
    name = "catch2_main",
    srcs = ["src/catch2/internal/catch_main.cpp"],
    includes = ["src/"],
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [":catch2"],
)
