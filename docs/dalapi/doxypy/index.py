# file: index.py
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

from typing import List, Text
from . import utils
from . import model
from .parser import Parser
from .loader import (
    ModelLoader,
    Transformer,
    NameTransformer,
    TransformerPass,
)

class _IndexEntry(object):
    def __init__(self, loader: ModelLoader, refid: Text):
        self._loader = loader
        self._refid = refid
        self._model = None

    @property
    def model(self):
        if not self._model:
            self._model = self._loader.load(self._refid)
            try:
                self._on_content_loaded(self._model)
            except AttributeError:
                pass
        return self._model


class Index(object):
    def __init__(self, parser: Parser, loader: ModelLoader,
                       name_transformer: NameTransformer = None):
        assert parser
        assert loader
        self._parser = parser
        self._loader = loader
        self._name_transformer = name_transformer
        self._index = self._initialize()

    def find(self, query: str):
        try:
            model = self._index[query].model
        except KeyError:
            try:
                model = self._find_inner(query)
            except KeyError:
                raise KeyError(f'Cannot find "{query}"')
        return model

    def find_all(self, query: str):
        try:
            models = self._find_inner_all(query)
        except KeyError:
            raise KeyError(f'Cannot find "{query}"')
        return models

    @utils.return_dict
    def _initialize(self):
        index = self._parser.parse('index')
        allowed_kinds = {'class', 'struct', 'namespace', 'typedef'}
        for entry in index.compound:
            if entry.kind in allowed_kinds:
                model_object = _IndexEntry(self._loader, entry.refid)
                name = entry.name
                if self._name_transformer:
                    name = self._name_transformer.transform(entry.name)
                yield name, model_object

    def _find_inner(self, query):
        parent_name, name = utils.split_compound_name(query)
        model = self._index[parent_name].model
        attrs_to_check = [ 'functions', 'typedefs', 'enum_classes' ]
        try:
            for attr in attrs_to_check:
                for inner in getattr(model, attr):
                    if inner.name == name:
                        return inner
        except AttributeError:
            pass
        raise KeyError(f'Cannot find {name} inside {parent_name}')

    def _find_inner_all(self, query):
        parent_name, name = utils.split_compound_name(query)
        model = self._index[parent_name].model
        attrs_to_check = [ 'functions', 'typedefs', 'enum_classes' ]
        try:
            for attr in attrs_to_check:
                result = []
                for inner in getattr(model, attr):
                    if inner.name == name:
                        result.append(inner)
                return result
        except AttributeError:
            pass
        raise KeyError(f'Cannot find {name} inside {parent_name}')

def index(doxygen_xml_dir: Text,
          name_transformer: NameTransformer = None,
          transformer_passes: List[TransformerPass] = []) -> Index:
    parser = Parser(doxygen_xml_dir)
    transformer = Transformer(name_transformer, transformer_passes)
    loader = ModelLoader(parser, transformer)
    return Index(parser, loader, name_transformer)

def to_dict(index: Index, discard_empty=False):
    index_dict = _to_dict(index)
    if discard_empty:
        index_dict = _discard_empty(index_dict)
    return index_dict

def to_json(index: Index, discard_empty=False, **kwargs):
    import json
    return json.dumps(to_dict(index, discard_empty), **kwargs)

def to_yaml(index: Index, discard_empty=False, **kwargs):
    import yaml
    return yaml.dump(to_dict(index, discard_empty), **kwargs)

@utils.return_dict
def _to_dict(index: Index):
    for k, v in index._index.items():
        yield k, _index_to_dict(v.model)

def _discard_empty(obj):
    if isinstance(obj, list):
        gen = (_discard_empty(x) for x in obj)
        obj = [x for x in gen if x]
    elif isinstance(obj, dict):
        gen = ((k, _discard_empty(x)) for k, x in obj.items())
        obj = {k: x for k, x in gen if x}
    if obj:
        return obj

def _index_to_dict(obj, discard_empty=False):
    if isinstance(obj, list):
        return [_index_to_dict(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _index_to_dict(x) for k, x in obj.items()}
    elif model.has_fields(obj):
        obj_dict = {}
        for field in obj.iter():
            value = getattr(obj, field)
            obj_dict[field] = _index_to_dict(value)
        return obj_dict
    return obj
