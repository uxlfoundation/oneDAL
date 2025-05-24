# file: directives.py
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

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList
from .generator import RstBuilder

def directive(cls):
    class DirectiveFactory(cls):
        def __init__(self, ctx):
            self._ctx = ctx
        def __call__(self, *args, **kwargs):
            return cls(self._ctx, *args, **kwargs)
    return DirectiveFactory

class BaseDirective(Directive):
    def __init__(self, ctx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctx = ctx

class MacroDirective(BaseDirective):
    def init(self): ...
    def rst(self, x: RstBuilder): ...

    def run(self):
        self.init()
        content = ViewList()
        self._get_rst(content)
        node = nodes.section()
        node.document = self.state.document
        self.state.nested_parse(content, 0, node)
        return node.children

    def _get_rst(self, placeholder):
        x = RstBuilder(placeholder, self.ctx.current_docname, self.lineno)
        self.rst(x)


class DoxyDirective(MacroDirective):
    def init(self):
        self.ctx.watcher.link_docname(self.ctx.current_docname)

    def format_description(self, desc_def):
        rst = ''
        for run in desc_def.runs:
            if run.kind == 'text':
                rst += run.content
            elif run.kind == 'math':
                rst += f':math:`{run.content}`'
            elif run.kind == 'code':
                if hasattr(run, 'directive'):
                    rst += f'{run.directive}`{run.content}`'
                else:
                    rst += f'`{run.content}`'
        return rst.strip()

    def add_description(self, description, x: RstBuilder, level=0):
        desc_str = self.format_description(description)
        if desc_str:
            x.add_doc(desc_str, level=level)

    def add_function_base(self, func, x: RstBuilder, is_free=True, level=0):
        if func.doc and func.doc.description:
            namespace = func.parent_fully_qualified_name if is_free else None
            x.add_function(func.declaration, namespace, level=level)
            self.add_description(func.doc.description, x, level=level + 1)
            self.add_params('tparam', func.template_parameters, x, level=level + 1)
            self.add_params('param', func.parameters, x, level=level + 1)
            x.add_blank_line()
            self.add_preconditions(func.doc.preconditions, x, level=level + 1)
            self.add_postconditions(func.doc.postconditions, x, level=level + 1)

    def add_params(self, tag, params, x: RstBuilder, level=0):
        for param in params:
            if param.description:
                desc_str = self.format_description(param.description)
                x.add_param(tag, param.name, desc_str, level=level)

    def add_listing(self, model_object, x: RstBuilder, remove_empty_lines=False):
        listing = self.ctx.listing.read(model_object, remove_empty_lines)
        x.add_code_block(listing)

    def add_invariants(self, invariants, x: RstBuilder, level=0):
        if invariants:
            x('Invariants', level=level)
            for invariant in invariants:
                x.add(f'| ' + self.format_description(invariant), level=level + 1)
            x.add_blank_line()

    def add_preconditions(self, preconditions, x: RstBuilder, level=0):
        if preconditions:
            x('Preconditions', level=level)
            for precondition in preconditions:
                x.add(f'| ' + self.format_description(precondition), level=level + 1)
            x.add_blank_line()

    def add_postconditions(self, postconditions, x: RstBuilder, level=0):
        if postconditions:
            x('Postconditions', level=level)
            for postcondition in postconditions:
                x.add(f'| ' + self.format_description(postcondition), level=level + 1)
            x.add_blank_line()


@directive
class ClassDirective(DoxyDirective):
    required_arguments = 1
    optional_arguments = 0
    has_content = False

    def rst(self, x: RstBuilder):
        class_ = self.ctx.index.find(self.arguments[0])
        if self.ctx.listing_enabled:
            self.add_listing(class_, x)
        sphinx_class_decl = (f'{class_.template_declaration} {class_.name}'
                             if class_.template_declaration else class_.name)
        x.add_class(class_.kind, sphinx_class_decl,
                    class_.parent_fully_qualified_name)
        self.add_params('tparam', class_.template_parameters, x, level=1)
        x.add_blank_line()
        self.add_static_methods(class_, x)
        self.add_constructors(class_, x)
        self.add_methods(class_, x)
        self.add_properties(class_, x)

    def add_static_methods(self, class_, x: RstBuilder):
        static_methods = [func for func in class_.static_functions
                          if func.return_type.strip()]
        if static_methods:
            x.add('**Public Static Methods**', level=1)
            x.add_blank_line()
        for method in static_methods:
            self.add_function_base(method, x, level=1)

    def add_constructors(self, class_, x: RstBuilder):
        ctors = [func for func in class_.functions
                 if not func.return_type.strip()]
        if ctors:
            x.add('**Constructors**', level=1)
            x.add_blank_line()
        for ctor in ctors:
            self.add_function_base(ctor, x, level=1)

    def add_methods(self, class_, x: RstBuilder):
        methods = [func for func in class_.functions
                   if func.return_type.strip()]
        if methods:
            x.add('**Public Methods**', level=1)
            x.add_blank_line()
        for method in methods:
            self.add_function_base(method, x, level=1)

    def add_properties(self, class_def, x: RstBuilder):
        if class_def.properties:
            x.add('**Properties**', level=1)
            x.add_blank_line()
        for property_def in class_def.properties:
            self.add_property(property_def, x)

    def add_property(self, property_def, x: RstBuilder):
        x.add_property_member(property_def.declaration,
                              property_def.parent_fully_qualified_name,
                              level=1)
        if property_def.doc and property_def.doc.description:
            desc = self.format_description(property_def.doc.description)
            if property_def.default:
                if not desc.endswith('.'):
                    desc += '.'
                desc += f' **Default value**: {property_def.default}'
            if desc:
                x.add_doc(desc, level=2)
            else:
                print(f'WARNING: oneDAL property `{property_def.fully_qualified_name}` is not documented')
        if property_def.getter or property_def.setter:
            x('Getter & Setter', level=2)
            if property_def.getter:
                x(f'| ``{property_def.getter.declaration}``', level=3)
            if property_def.setter:
                x(f'| ``{property_def.setter.declaration}``', level=3)
            x()
        if property_def.doc:
            invariants = property_def.doc.invariants
            self.add_invariants(invariants, x, level=2)


@directive
class FunctionDirective(DoxyDirective):
    required_arguments = 1
    optional_arguments = 0
    has_content = False

    def rst(self, x: RstBuilder):
        # Find all functions with the same name in the index and add them to the documentation
        funcs = self.ctx.index.find_all(self.arguments[0])
        for func in funcs:
            if self.ctx.listing_enabled:
                self.add_listing(func, x)
            self.add_function_base(func, x, is_free=True)

@directive
class EnumClassDirective(DoxyDirective):
    required_arguments = 1
    optional_arguments = 0
    has_content = False

    def rst(self, x: RstBuilder):
        enum = self.ctx.index.find(self.arguments[0])
        namespace = enum.parent_fully_qualified_name

        if self.ctx.listing_enabled:
            self.add_listing(enum, x)
        x.add_blank_line()
        x.add_enumclass(enum.name, namespace)

        for value in enum.values:
            x.add(f'{enum.name}::{value.name}', level=1)

            if value.doc and value.doc.description:
                self.add_description(value.doc.description, x, level=2)


@directive
class ListingDirective(DoxyDirective):
    required_arguments = 1
    optional_arguments = 0
    has_content = False

    def rst(self, x: RstBuilder):
        model_object = self.ctx.index.find(self.arguments[0])
        listing = self.ctx.listing.read(model_object)
        x.add_code_block(listing)


@directive
class TagsNamespaceDirective(DoxyDirective):
    required_arguments = 1
    optional_arguments = 0
    has_content = False

    def rst(self, x: RstBuilder):
        methods_namespace = self.arguments[0]
        method_ns = self.ctx.index.find(methods_namespace)
        if self.ctx.listing_enabled:
            self.add_listing(method_ns, x, remove_empty_lines=True)
        self._add_classes(method_ns, x)
        self._add_typedefs(method_ns, x)

    def _add_classes(self, method_ns, x: RstBuilder):
        for class_ref in method_ns.class_refs:
            class_fqn = class_ref.fully_qualified_name
            class_ = self.ctx.index.find(class_fqn)
            x.add_class(class_.kind, class_.name, class_.parent_fully_qualified_name)
            if class_.doc and class_.doc.description:
                self.add_description(class_.doc.description, x, level=1)

    def _add_typedefs(self, method_ns, x: RstBuilder):
        for typedef in method_ns.typedefs:
            sphinx_decl = f'{typedef.name} = {typedef.type}'
            x.add_typedef(sphinx_decl, typedef.parent_fully_qualified_name)
            if typedef.doc and typedef.doc.description:
                self.add_description(typedef.doc.description, x, level=1)
