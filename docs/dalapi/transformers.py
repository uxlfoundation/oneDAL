# file: transformers.py
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

import re
from typing import (Dict, Tuple, Text)
from collections import OrderedDict, namedtuple
from . import doxypy

class NameTransformer(doxypy.NameTransformer):
    _v_namespace_re = re.compile(r'::v\d+')

    def transform(self, fully_qualified_name):
        v_namespace_re = NameTransformer._v_namespace_re
        return v_namespace_re.sub('', fully_qualified_name)


@doxypy.model.model_object
class Property(object):
    doc: doxypy.model.Doc = None
    name: Text = None
    type: Text = None
    getter: doxypy.model.Function = None
    setter: doxypy.model.Function = None
    default: Text = None
    declaration: Text = None
    fully_qualified_name: Text = None
    parent_fully_qualified_name: Text = None

class PropertyTransformer(doxypy.TransformerPass):
    _accessor_re = re.compile(r'(get|set)_(\w+)')
    _default_re = re.compile(r'default *= *(.+)')

    def enter(self, node):
        return (isinstance(node, (doxypy.model.Class, doxypy.model.Namespace)))

    def transform(self, node):
        if isinstance(node, doxypy.model.Class):
            properties = []
            for name, info in self._get_properties_info(node):
                prop = self._build_property(name, info)
                prop.getter.doc = None
                node.functions.remove(prop.getter)
                if prop.setter:
                    node.functions.remove(prop.setter)
                properties.append(prop)
            node.properties = properties

    @classmethod
    def _build_property(cls, name, info):
        parent_fqn = f'{info.getter.parent_fully_qualified_name}::{name}'
        default = cls._find_default(info)
        decl = f'{info.getter.return_type} {name}'
        return Property(
            doc = info.getter.doc,
            name = name,
            type = info.getter.return_type,
            getter = info.getter,
            setter = info.setter,
            default = default,
            declaration = decl,
            fully_qualified_name = parent_fqn,
            parent_fully_qualified_name = info.getter.parent_fully_qualified_name,
        )

    @classmethod
    def _find_default(cls, info):
        if info.getter.doc:
            for remark in info.getter.doc.remarks:
                if len(remark.runs) == 1 and remark.runs[0].kind == 'text':
                    match = cls._default_re.match(remark.runs[0].content)
                    if match:
                        return match.group(1)

    @classmethod
    def _get_properties_info(cls, node):
        getters = OrderedDict(cls._get_access_methods(node, 'get'))
        setters = OrderedDict(cls._get_access_methods(node, 'set'))
        PropertyInfo = namedtuple('PropertyInfo', ['getter', 'setter'])
        for name, getter in getters.items():
            if name in setters:
                yield name, PropertyInfo(getter, setters[name])

    @classmethod
    def _get_access_methods(cls, node, direction):
        for func in node.functions:
            match = cls._accessor_re.match(func.name)
            if match and match.group(1) == direction:
                yield match.group(2), func


class RstDescriptionTransformer(doxypy.TransformerPass):
    _sphinx_directive_re = re.compile(r':([\w:]+):$')

    def enter(self, node):
        return True

    def transform(self, node):
        if isinstance(node, doxypy.model.Description):
            self._transform_runs(node)

    @classmethod
    def _transform_runs(cls, desc: doxypy.model.Description):
        for i in range(1, len(desc.runs)):
            pre_run = desc.runs[i - 1]
            cur_run = desc.runs[i]
            if pre_run.kind == 'text' and cur_run.kind == 'code':
                new_content, directive = cls._try_remove_sphinx_directive(pre_run.content)
                if directive:
                    pre_run.content = new_content
                    cur_run.directive = cls._map_directive(directive)

    @classmethod
    def _try_remove_sphinx_directive(cls, text):
        match = None
        def rep_f(m):
            nonlocal match
            match = m
            return ''
        new_text = cls._sphinx_directive_re.sub(rep_f, text)
        return new_text, match.group(1) if match else None

    @classmethod
    def _map_directive(cls, directive):
        cpp_directives = { 'expr', 'texpr', 'any', 'class',
                           'struct', 'func', 'member', 'var',
                           'type', 'concept', 'enum', 'enumerator' }
        if directive in cpp_directives:
            return f':cpp:{directive}:'
        return f':{directive}:'
