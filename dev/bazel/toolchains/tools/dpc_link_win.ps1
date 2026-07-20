#===============================================================================
# Copyright 2026 Intel Corporation
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

<#
.SYNOPSIS
Adapts Bazel's Windows DPC++ link arguments for the Intel compiler driver.

.DESCRIPTION
The generated `dpc_link_win.bat` launcher invokes this committed script for
DPC++ link actions. The Intel compiler driver must remain in the link path so
it can add the SYCL device-code libraries, but very large Bazel link actions
can exceed Windows response-file line limits when all object inputs are passed
through the driver unchanged.

The wrapper expands Bazel response files, moves `.obj` and `.res` inputs into
a temporary response file with one argument per line, and inserts that file at
the position of the first object input. Non-object arguments retain their
original order. The compiler's exit code is propagated and the temporary file
is removed even when linking fails.

.PARAMETER Compiler
Path to the Intel DPC++ compiler driver substituted by toolchain setup.

.PARAMETER RawArgs
The remaining arguments from the Bazel link action.
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Compiler,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RawArgs
)

$expandedArgs = New-Object System.Collections.Generic.List[string]
foreach ($arg in $RawArgs) {
    if ($arg.StartsWith('@')) {
        # Bazel writes one argument per line in these response files.
        $path = $arg.Substring(1)
        foreach ($line in [IO.File]::ReadLines($path)) {
            if ($line.Length -gt 0) {
                $expandedArgs.Add($line)
            }
        }
    }
    else {
        $expandedArgs.Add($arg)
    }
}

$objectRsp = Join-Path ([IO.Path]::GetTempPath()) ([IO.Path]::GetRandomFileName() + '.rsp')
$objectArgs = New-Object System.Collections.Generic.List[string]
$finalArgs = New-Object System.Collections.Generic.List[string]
$insertedObjectRsp = $false

foreach ($arg in $expandedArgs) {
    if ($arg -match '(?i)\.(obj|res)$') {
        $objectArgs.Add($arg)
        if (-not $insertedObjectRsp) {
            $finalArgs.Add('@' + $objectRsp)
            $insertedObjectRsp = $true
        }
    }
    else {
        $finalArgs.Add($arg)
    }
}

try {
    # An empty file is harmless for actions without object/resource inputs.
    [IO.File]::WriteAllLines($objectRsp, $objectArgs)
    & $Compiler @finalArgs
    exit $LASTEXITCODE
}
finally {
    Remove-Item -Force $objectRsp -ErrorAction SilentlyContinue
}
