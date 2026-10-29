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

# find latest version at https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html
echo "Download intel opencl runtime"
(new-object System.Net.WebClient).DownloadFile("https://registrationcenter-download.intel.com/akdlm/IRC_NAS/f169cc87-7163-4df5-94d2-bfd07d42b204/w_opencl_runtime_p_2026.0.0.946.exe", "opencl_installer.exe")
echo "Unpacking opencl runtime installer"
Start-Process ".\opencl_installer.exe" -ArgumentList "--s --x --f ocl" -Wait
Move-Item -Path ".\ocl\w_opencl_runtime_p_2025.3.1.762.msi" -Destination ".\opencl_rt.msi"
