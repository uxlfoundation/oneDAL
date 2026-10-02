@echo off
rem ============================================================================
rem Copyright 2018 Intel Corporation
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.
rem ============================================================================

rem Exit non-zero: this stands in for a tool the toolchain could not find, so
rem every action that reaches it has to fail. Succeeding without writing the
rem declared outputs makes Bazel report "not all outputs were created" and
rem hides the message below.
echo %{tool_name} is not found! Make sure it is available in PATH. 1>&2
exit /b 1
