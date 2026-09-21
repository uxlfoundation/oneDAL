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

"""Central selects for the BUILD_PARAMETERS_LIB layout."""

_FOLDED = [
    "@config//:build_parameters_lib_auto_windows",
    "@config//:build_parameters_lib_yes_windows",
    "@config//:build_parameters_lib_no",
]

def parameters_lib_enabled(values):
    """Values for separate layout: auto on non-Windows, or explicit yes."""
    choices = {setting: [] for setting in _FOLDED}
    choices["//conditions:default"] = values
    return select(choices)

def parameters_lib_disabled(values):
    """Values for folded layout: no, and auto/unsupported yes on Windows."""
    choices = {setting: values for setting in _FOLDED}
    choices["//conditions:default"] = []
    return select(choices)

def parameters_lib_target_compatible_with():
    """Reject public parameter-library targets when the layout is folded."""
    choices = {setting: ["@platforms//:incompatible"] for setting in _FOLDED}
    choices["//conditions:default"] = []
    return select(choices)

def parameters_lib_separate_value():
    """Boolean metadata value for the selected release layout."""
    choices = {setting: False for setting in _FOLDED}
    choices["//conditions:default"] = True
    return select(choices)

def release_dpc_parameters_lib_enabled(values):
    """Values when DPC release and separate parameters are both enabled."""
    choices = {setting: [] for setting in _FOLDED}
    choices["@config//:release_dpc_disabled"] = []
    choices["//conditions:default"] = values
    return select(choices)
