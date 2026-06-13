# file: generator.py
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

from typing import (List, Text)
from docutils.statemachine import ViewList

class RstBuilder(object):
    def __init__(self, placeholder: ViewList, filename: Text, lineno: int):
        self._rst_list = placeholder
        self._filename = filename
        self._lineno = lineno

    def add_class(self, kind: str, declaration: str, namespace: str = None, level=0):
        self._add_name(kind, declaration, namespace, level)

    def add_typedef(self, declaration: str, namespace: str = None, level=0):
        self._add_name('type', declaration, namespace, level)

    def add_function(self, declaration: str, namespace: str = None, level=0):
        self._add_name('function', declaration, namespace, level)

    def add_enumclass(self, declaration: str, namespace: str = None, level=0):
        self._add_name('enum-class', declaration, namespace, level)

    def add_param(self, tag: str, name: str, doc_text: str, level=0):
        assert tag in ['param', 'tparam']
        assert name
        assert doc_text
        formatted = self._format_text(doc_text)
        self(f':{tag} {name}: {formatted}', level)

    def add_member(self, declaration: str, level=0):
        assert declaration
        self(f'.. cpp:member:: {declaration}', level)
        self.add_blank_line()

    def add_property_member(self, declaration: str, parent_fully_qualified_name: str, level=0):
        assert declaration
        assert parent_fully_qualified_name
        self(f'.. cpp:member:: {declaration}', level)
        self.add_blank_line()

    def add_doc(self, doc_text: str, level=0):
        assert doc_text
        self(self._format_text(doc_text), level)
        self.add_blank_line()

    def add_code_block(self, listing: List[Text], level=0):
        assert listing is not None
        self('.. code-block:: cpp', level)
        self.add_blank_line()
        for line in listing:
            self(line, level + 1)
        self.add_blank_line()

    def add_blank_line(self):
        self.add()

    def add(self, string: str = '', level: int = 0):
        self._rst_list.append(' ' * level * 3 + string, self._filename, self._lineno)

    # Keep callable behavior for existing builder call sites.
    def __call__(self, string: str = '', level:int = 0):
        self._rst_list.append(' ' * level * 3 + string, self._filename, self._lineno)

    def _add_name(self, tag: str, declaration: str, namespace: str = None, level=0):
        assert declaration
        if namespace:
            self(f'.. cpp:namespace:: {namespace}', level)
            self.add_blank_line()
        self(f'.. cpp:{tag}:: {declaration}', level)
        self.add_blank_line()

    def _format_text(self, text):
        text = text.strip()
        if not (text.endswith('.') or text.endswith('|')):
            text += '.'
        return text
