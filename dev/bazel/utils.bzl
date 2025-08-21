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

load("@bazel_skylib//lib:paths.bzl", _paths = "paths")
load("@bazel_skylib//lib:new_sets.bzl", _sets = "sets")
load("@bazel_skylib//lib:collections.bzl", _collections = "collections")

paths = _paths
sets = _sets
collections = _collections

# Collection functions
# ====================

def _unique(iterable):
    """Remove duplicates from a list."""
    return collections.uniq(iterable)

def _unique_files(files):
    files_dict = {f.path: f for f in files }
    return files_dict.values()

def _normalize_dict(dict_to_normalizes, mandatory_keys, default=None):
    normalized_dict = {}
    for key in mandatory_keys:
        normalized_dict[key] = dict_to_normalizes.get(key, default)
    return normalized_dict

def _filter_out(lst_to_filter, filter):
    filter_set = sets.make(filter)
    filtered = []
    for item in lst_to_filter:
        if not sets.contains(filter_set, item):
            filtered.append(item)
    return filtered


# String functions
# ================

def _add_prefix(prefix, lst):
    return [ prefix + str(x) for x in lst ]

def _substitute(string, substitutions={}):
    string_fmt = string
    for key, value in substitutions.items():
        string_fmt = string_fmt.replace(key, value)
    return string_fmt

def _match_substring(string, substrings):
    for substring in substrings:
        if string.rfind(substring) > 0:
            return substring

def _remove_substring(string, substring):
    index = string.rfind(substring)
    if index > 0:
        return string[:index] + string[index + len(substring):]
    else:
        return string

# Output functions
# ================

def _warn(msg):
    """Output warning."""
    yellow = "\033[1;33m"
    no_color = "\033[0m"
    print("\n%sWARNING:%s %s\n" % (yellow, no_color, msg))

def _info(msg):
    """Output warning."""
    yellow = "\033[0;32m"
    no_color = "\033[0m"
    print("\n%sINFO:%s %s\n" % (yellow, no_color, msg))

# Other functions
# ===============

def _datestamp(repo_ctx):
    return repo_ctx.execute(["date", "+%Y%m%d"]).stdout.strip()

utils = struct(
    unique = _unique,
    unique_files = _unique_files,
    normalize_dict = _normalize_dict,
    filter_out = _filter_out,
    add_prefix = _add_prefix,
    substitute = _substitute,
    match_substring = _match_substring,
    remove_substring = _remove_substring,
    warn = _warn,
    info = _info,
    datestamp = _datestamp,
)
