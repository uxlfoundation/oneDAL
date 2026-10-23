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

"""Product and library binary versions of the Bazel build.

The one place to bump them. `makefile.ver` carries the same values for the
Make build until it is removed.
"""

VERSION = struct(
    major = "2026",
    minor = "3",
    update = "0",
    status = "P",
    binary_major = "4",
    binary_minor = "0",
)
