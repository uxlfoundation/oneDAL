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

param(
    [Parameter(Mandatory=$true)]
    [string]$Compiler,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RawArgs
)

$expandedArgs = New-Object System.Collections.Generic.List[string]
foreach ($arg in $RawArgs) {
    if ($arg.StartsWith('@')) {
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
    [IO.File]::WriteAllLines($objectRsp, $objectArgs)
    & $Compiler @finalArgs
    exit $LASTEXITCODE
}
finally {
    Remove-Item -Force $objectRsp -ErrorAction SilentlyContinue
}
