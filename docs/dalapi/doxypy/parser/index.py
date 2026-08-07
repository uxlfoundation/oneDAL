#!/usr/bin/env python
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

# -*- coding: utf-8 -*-

#
# Generated Fri Apr 24 17:42:25 2020 by generateDS.py version 2.35.21.
# Python 3.6.10 |Anaconda, Inc.| (default, Mar 25 2020, 23:51:54)  [GCC 7.3.0]
#
# Command line options:
#   ('-o', '../../dalapi/index_parser.py')
#
# Command line arguments:
#   index.xsd
#
# Command line:
#   /root/miniconda3/envs/spec/bin/generateDS -o "../../dalapi/index_parser.py" index.xsd
#
# Current working directory (os.getcwd()):
#   xml
#

import sys

from .common import (
    CDATA_pattern_,
    CapturedNsmap_,
    CurrentSubclassModule_,
    Enum,
    ExternalEncoding,
    GDSParseError,
    GenerateDSNamespaceDefs_,
    GenerateDSNamespaceTypePrefixes_,
    GeneratedsSuper,
    GdsCollector_,
    MemberSpec_,
    MixedContainer,
    Namespace_extract_pat_,
    SaveElementTreeNode,
    String_cleanup_pat_,
    Tag_pattern_,
    UseCapturedNS_,
    Validate_simpletypes_,
    _cast,
    encode_str_2_3,
    etree_,
    find_attr_value_,
    get_all_text_,
    getSubclassFromModule_,
    parsexml_,
    parsexmlstring_,
    quote_attrib,
    quote_python,
    quote_xml,
    quote_xml_aux,
    raise_parse_error,
    showIndent,
)

#
# Data representation classes.
#


class CompoundKind(Enum):
    CLASS='class'
    STRUCT='struct'
    UNION='union'
    INTERFACE='interface'
    PROTOCOL='protocol'
    CATEGORY='category'
    EXCEPTION='exception'
    FILE='file'
    NAMESPACE='namespace'
    GROUP='group'
    PAGE='page'
    EXAMPLE='example'
    DIR='dir'


class MemberKind(Enum):
    DEFINE='define'
    PROPERTY='property'
    EVENT='event'
    VARIABLE='variable'
    TYPEDEF='typedef'
    ENUM='enum'
    ENUMVALUE='enumvalue'
    FUNCTION='function'
    SIGNAL='signal'
    PROTOTYPE='prototype'
    FRIEND='friend'
    DCOP='dcop'
    SLOT='slot'


class DoxygenType(GeneratedsSuper):
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, version=None, compound=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.version = _cast(None, version)
        self.version_nsprefix_ = None
        if compound is None:
            self.compound = []
        else:
            self.compound = compound
        self.compound_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, DoxygenType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if DoxygenType.subclass:
            return DoxygenType.subclass(*args_, **kwargs_)
        return DoxygenType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_compound(self):
        return self.compound
    def set_compound(self, compound):
        self.compound = compound
    def add_compound(self, value):
        self.compound.append(value)
    def insert_compound_at(self, index, value):
        self.compound.insert(index, value)
    def replace_compound_at(self, index, value):
        self.compound[index] = value
    def get_version(self):
        return self.version
    def set_version(self, version):
        self.version = version
    def hasContent_(self):
        if (
            self.compound
        ):
            return True
        return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='DoxygenType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('DoxygenType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'DoxygenType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='DoxygenType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='DoxygenType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='DoxygenType'):
        if self.version is not None and 'version' not in already_processed:
            already_processed.add('version')
            outfile.write(' version=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.version), input_name='version')), ))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='DoxygenType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        for compound_ in self.compound:
            namespaceprefix_ = self.compound_nsprefix_ + ':' if (UseCapturedNS_ and self.compound_nsprefix_) else ''
            compound_.export(outfile, level, namespaceprefix_, namespacedef_='', name_='compound', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('version', node)
        if value is not None and 'version' not in already_processed:
            already_processed.add('version')
            self.version = value
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'compound':
            obj_ = CompoundType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.compound.append(obj_)
            obj_.original_tagname_ = 'compound'
# end class DoxygenType


class CompoundType(GeneratedsSuper):
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, refid=None, kind=None, name=None, member=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.refid = _cast(None, refid)
        self.refid_nsprefix_ = None
        self.kind = _cast(None, kind)
        self.kind_nsprefix_ = None
        self.name = name
        self.name_nsprefix_ = None
        if member is None:
            self.member = []
        else:
            self.member = member
        self.member_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, CompoundType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if CompoundType.subclass:
            return CompoundType.subclass(*args_, **kwargs_)
        return CompoundType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name
    def get_member(self):
        return self.member
    def set_member(self, member):
        self.member = member
    def add_member(self, value):
        self.member.append(value)
    def insert_member_at(self, index, value):
        self.member.insert(index, value)
    def replace_member_at(self, index, value):
        self.member[index] = value
    def get_refid(self):
        return self.refid
    def set_refid(self, refid):
        self.refid = refid
    def get_kind(self):
        return self.kind
    def set_kind(self, kind):
        self.kind = kind
    def validate_CompoundKind(self, value):
        # Validate type CompoundKind, a restriction on xsd:string.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            value = value
            enumerations = ['class', 'struct', 'union', 'interface', 'protocol', 'category', 'exception', 'file', 'namespace', 'group', 'page', 'example', 'dir']
            if value not in enumerations:
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s does not match xsd enumeration restriction on CompoundKind' % {"value" : encode_str_2_3(value), "lineno": lineno} )
                result = False
    def hasContent_(self):
        if (
            self.name is not None or
            self.member
        ):
            return True
        return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='CompoundType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('CompoundType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'CompoundType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='CompoundType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='CompoundType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='CompoundType'):
        if self.refid is not None and 'refid' not in already_processed:
            already_processed.add('refid')
            outfile.write(' refid=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.refid), input_name='refid')), ))
        if self.kind is not None and 'kind' not in already_processed:
            already_processed.add('kind')
            outfile.write(' kind=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.kind), input_name='kind')), ))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='CompoundType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.name is not None:
            namespaceprefix_ = self.name_nsprefix_ + ':' if (UseCapturedNS_ and self.name_nsprefix_) else ''
            showIndent(outfile, level, pretty_print)
            outfile.write('<%sname>%s</%sname>%s' % (namespaceprefix_ , self.gds_encode(self.gds_format_string(quote_xml(self.name), input_name='name')), namespaceprefix_ , eol_))
        for member_ in self.member:
            namespaceprefix_ = self.member_nsprefix_ + ':' if (UseCapturedNS_ and self.member_nsprefix_) else ''
            member_.export(outfile, level, namespaceprefix_, namespacedef_='', name_='member', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('refid', node)
        if value is not None and 'refid' not in already_processed:
            already_processed.add('refid')
            self.refid = value
        value = find_attr_value_('kind', node)
        if value is not None and 'kind' not in already_processed:
            already_processed.add('kind')
            self.kind = value
            self.validate_CompoundKind(self.kind)    # validate type CompoundKind
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'name':
            value_ = child_.text
            value_ = self.gds_parse_string(value_, node, 'name')
            value_ = self.gds_validate_string(value_, node, 'name')
            self.name = value_
            self.name_nsprefix_ = child_.prefix
        elif nodeName_ == 'member':
            obj_ = MemberType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.member.append(obj_)
            obj_.original_tagname_ = 'member'
# end class CompoundType


class MemberType(GeneratedsSuper):
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, refid=None, kind=None, name=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.refid = _cast(None, refid)
        self.refid_nsprefix_ = None
        self.kind = _cast(None, kind)
        self.kind_nsprefix_ = None
        self.name = name
        self.name_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, MemberType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if MemberType.subclass:
            return MemberType.subclass(*args_, **kwargs_)
        return MemberType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name
    def get_refid(self):
        return self.refid
    def set_refid(self, refid):
        self.refid = refid
    def get_kind(self):
        return self.kind
    def set_kind(self, kind):
        self.kind = kind
    def validate_MemberKind(self, value):
        # Validate type MemberKind, a restriction on xsd:string.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            value = value
            enumerations = ['define', 'property', 'event', 'variable', 'typedef', 'enum', 'enumvalue', 'function', 'signal', 'prototype', 'friend', 'dcop', 'slot']
            if value not in enumerations:
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s does not match xsd enumeration restriction on MemberKind' % {"value" : encode_str_2_3(value), "lineno": lineno} )
                result = False
    def hasContent_(self):
        if (
            self.name is not None
        ):
            return True
        return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='MemberType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('MemberType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'MemberType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='MemberType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='MemberType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='MemberType'):
        if self.refid is not None and 'refid' not in already_processed:
            already_processed.add('refid')
            outfile.write(' refid=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.refid), input_name='refid')), ))
        if self.kind is not None and 'kind' not in already_processed:
            already_processed.add('kind')
            outfile.write(' kind=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.kind), input_name='kind')), ))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='MemberType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.name is not None:
            namespaceprefix_ = self.name_nsprefix_ + ':' if (UseCapturedNS_ and self.name_nsprefix_) else ''
            showIndent(outfile, level, pretty_print)
            outfile.write('<%sname>%s</%sname>%s' % (namespaceprefix_ , self.gds_encode(self.gds_format_string(quote_xml(self.name), input_name='name')), namespaceprefix_ , eol_))
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('refid', node)
        if value is not None and 'refid' not in already_processed:
            already_processed.add('refid')
            self.refid = value
        value = find_attr_value_('kind', node)
        if value is not None and 'kind' not in already_processed:
            already_processed.add('kind')
            self.kind = value
            self.validate_MemberKind(self.kind)    # validate type MemberKind
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'name':
            value_ = child_.text
            value_ = self.gds_parse_string(value_, node, 'name')
            value_ = self.gds_validate_string(value_, node, 'name')
            self.name = value_
            self.name_nsprefix_ = child_.prefix
# end class MemberType


GDSClassesMapping = {
    'doxygenindex': DoxygenType,
}


USAGE_TEXT = """
Usage: python <Parser>.py [ -s ] <in_xml_file>
"""


def usage():
    print(USAGE_TEXT)
    sys.exit(1)


def get_root_tag(node):
    tag = Tag_pattern_.match(node.tag).groups()[-1]
    rootClass = GDSClassesMapping.get(tag)
    if rootClass is None:
        rootClass = globals().get(tag)
    return tag, rootClass


def get_required_ns_prefix_defs(rootNode):
    '''Get all name space prefix definitions required in this XML doc.
    Return a dictionary of definitions and a char string of definitions.
    '''
    nsmap = {
        prefix: uri
        for node in rootNode.iter()
        for (prefix, uri) in node.nsmap.items()
        if prefix is not None
    }
    namespacedefs = ' '.join([
        'xmlns:{}="{}"'.format(prefix, uri)
        for prefix, uri in nsmap.items()
    ])
    return nsmap, namespacedefs


def parse(inFileName, silence=False, print_warnings=True):
    global CapturedNsmap_
    gds_collector = GdsCollector_()
    parser = None
    doc = parsexml_(inFileName, parser)
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'DoxygenType'
        rootClass = DoxygenType
    rootObj = rootClass.factory()
    rootObj.build(rootNode, gds_collector_=gds_collector)
    CapturedNsmap_, namespacedefs = get_required_ns_prefix_defs(rootNode)
    if not SaveElementTreeNode:
        doc = None
        rootNode = None
    if not silence:
        sys.stdout.write('<?xml version="1.0" ?>\n')
        rootObj.export(
            sys.stdout, 0, name_=rootTag,
            namespacedef_=namespacedefs,
            pretty_print=True)
    if print_warnings and len(gds_collector.get_messages()) > 0:
        separator = ('-' * 50) + '\n'
        sys.stderr.write(separator)
        sys.stderr.write('----- Warnings -- count: {} -----\n'.format(
            len(gds_collector.get_messages()), ))
        gds_collector.write_messages(sys.stderr)
        sys.stderr.write(separator)
    return rootObj


def parseEtree(inFileName, silence=False, print_warnings=True,
               mapping=None, nsmap=None):
    parser = None
    doc = parsexml_(inFileName, parser)
    gds_collector = GdsCollector_()
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'DoxygenType'
        rootClass = DoxygenType
    rootObj = rootClass.factory()
    rootObj.build(rootNode, gds_collector_=gds_collector)
    # Enable Python to collect the space used by the DOM.
    if mapping is None:
        mapping = {}
    rootElement = rootObj.to_etree(
        None, name_=rootTag, mapping_=mapping, nsmap_=nsmap)
    reverse_mapping = rootObj.gds_reverse_node_mapping(mapping)
    if not SaveElementTreeNode:
        doc = None
        rootNode = None
    if not silence:
        content = etree_.tostring(
            rootElement, pretty_print=True,
            xml_declaration=True, encoding="utf-8")
        sys.stdout.write(str(content))
        sys.stdout.write('\n')
    if print_warnings and len(gds_collector.get_messages()) > 0:
        separator = ('-' * 50) + '\n'
        sys.stderr.write(separator)
        sys.stderr.write('----- Warnings -- count: {} -----\n'.format(
            len(gds_collector.get_messages()), ))
        gds_collector.write_messages(sys.stderr)
        sys.stderr.write(separator)
    return rootObj, rootElement, mapping, reverse_mapping


def parseString(inString, silence=False, print_warnings=True):
    '''Parse a string, create the object tree, and export it.

    Arguments:
    - inString -- A string.  This XML fragment should not start
      with an XML declaration containing an encoding.
    - silence -- A boolean.  If False, export the object.
    Returns -- The root object in the tree.
    '''
    parser = None
    rootNode= parsexmlstring_(inString, parser)
    gds_collector = GdsCollector_()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'DoxygenType'
        rootClass = DoxygenType
    rootObj = rootClass.factory()
    rootObj.build(rootNode, gds_collector_=gds_collector)
    if not SaveElementTreeNode:
        rootNode = None
    if not silence:
        sys.stdout.write('<?xml version="1.0" ?>\n')
        rootObj.export(
            sys.stdout, 0, name_=rootTag,
            namespacedef_='')
    if print_warnings and len(gds_collector.get_messages()) > 0:
        separator = ('-' * 50) + '\n'
        sys.stderr.write(separator)
        sys.stderr.write('----- Warnings -- count: {} -----\n'.format(
            len(gds_collector.get_messages()), ))
        gds_collector.write_messages(sys.stderr)
        sys.stderr.write(separator)
    return rootObj


def parseLiteral(inFileName, silence=False, print_warnings=True):
    parser = None
    doc = parsexml_(inFileName, parser)
    gds_collector = GdsCollector_()
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'DoxygenType'
        rootClass = DoxygenType
    rootObj = rootClass.factory()
    rootObj.build(rootNode, gds_collector_=gds_collector)
    # Enable Python to collect the space used by the DOM.
    if not SaveElementTreeNode:
        doc = None
        rootNode = None
    if not silence:
        sys.stdout.write('#from index_parser import *\n\n')
        sys.stdout.write('import index_parser as model_\n\n')
        sys.stdout.write('rootObj = model_.rootClass(\n')
        rootObj.exportLiteral(sys.stdout, 0, name_=rootTag)
        sys.stdout.write(')\n')
    if print_warnings and len(gds_collector.get_messages()) > 0:
        separator = ('-' * 50) + '\n'
        sys.stderr.write(separator)
        sys.stderr.write('----- Warnings -- count: {} -----\n'.format(
            len(gds_collector.get_messages()), ))
        gds_collector.write_messages(sys.stderr)
        sys.stderr.write(separator)
    return rootObj


def main():
    args = sys.argv[1:]
    if len(args) == 1:
        parse(args[0])
    else:
        usage()


if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    main()

RenameMappings_ = {
}

__all__ = [
    "CompoundType",
    "DoxygenType",
    "MemberType"
]
