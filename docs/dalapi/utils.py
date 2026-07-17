# file: utils.py
#===============================================================================
# Copyright 2019 Intel Corporation
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

import os
import shlex
import subprocess
from typing import (
    Iterable,
    Sequence,
    Union,
)
from glob import iglob

class _cd:
    def __init__(self, new_path):
        self.new_path = os.path.abspath(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)


class ProcessHandle(object):
    def __init__(self, command: Union[str, Sequence[str]], startup_dir: str = '.'):
        self._command = command
        self._startup_dir = startup_dir

    def run(self):
        with _cd(self._startup_dir):
            command = (
                shlex.split(self._command)
                if isinstance(self._command, str)
                else self._command
            )
            subprocess.check_call(command)


class FileModificationTimer(object):
    def __init__(self, base_dir_or_files: Union[str, Iterable[str]],
                       pattern: str = '*'):
        if isinstance(base_dir_or_files, str):
            self._base_dir = os.path.abspath(base_dir_or_files)
            self._pattern = pattern
        else:
            self._files = base_dir_or_files

    def __call__(self):
        mtimes = [os.path.getmtime(x) for x in self._get_files()]
        return max(mtimes) if len(mtimes) > 0 else 0

    def _get_files(self):
        if hasattr(self, '_base_dir'):
            glob_str = f'{self._base_dir}/**/{self._pattern}'
            return iglob(glob_str, recursive=True)
        return self._files
