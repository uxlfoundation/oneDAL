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

param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$OutputPath,
    [string]$Cpus = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Cpus)) {
    Copy-Item -LiteralPath $InputPath -Destination $OutputPath -Force
    exit 0
}

$cpuPattern = ($Cpus -split '\s+' | Where-Object { $_ } | ForEach-Object { [regex]::Escape($_.ToUpperInvariant()) }) -join '|'
$pattern = "(?m)^#define DAAL_KERNEL_($cpuPattern)\b"

# Match the makefile byte for byte. Its recipe is
#   sed -b -i -E -e 's/^#define DAAL_KERNEL_<CPU>\b/$(sed.eol)/'
# with `sed.eol.win = \r`, so a disabled define becomes a lone CR that turns the
# existing LF into CRLF, while every other line keeps its original LF ending.
# Line-oriented cmdlets cannot express that: `Get-Content` drops the endings and
# `Set-Content` re-emits all of them as CRLF, which made the released header
# differ from the Make one on every line.
$content = [System.IO.File]::ReadAllText($InputPath)
$patched = [System.Text.RegularExpressions.Regex]::Replace($content, $pattern, "`r")
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($OutputPath, $patched, $utf8NoBom)
