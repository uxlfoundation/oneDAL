#!/bin/bash
#===============================================================================
# Copyright 2014 Intel Corporation
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

# Exit non-zero: this stands in for a tool the toolchain could not find, so
# every action that reaches it has to fail. Succeeding without writing the
# declared outputs makes Bazel report "not all outputs were created" and hides
# the message below.
echo "%{tool_name} is not found!" >&2
echo "Make sure %{tool_name} is available in \$PATH" >&2
exit 1
