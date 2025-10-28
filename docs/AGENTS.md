
# Documentation Generation - AI Agents Context

> **Purpose**: Context for AI agents working with oneDAL's sophisticated documentation generation system and customizations.

## 🏗️ Documentation Architecture

oneDAL implements a **sophisticated multi-stage documentation system** combining multiple technologies:

### Core Technology Stack
- **Sphinx**: Primary documentation engine with reStructuredText
- **Doxygen**: C++ API documentation extraction  
- **Custom dalapi Extension**: Bridge between Doxygen and Sphinx
- **Custom Python Tools**: RST generation and processing

### Build Pipeline
1. **RST Generation**: `create_rst_examples` → Dynamic example RST files
2. **Doxygen Processing**: `doxygen` → XML API extraction from C++ headers
3. **Doxygen Parsing**: `parse-doxygen` → Custom Python parser creates structured YAML
4. **Sphinx Build**: Multiple targets with different configurations

## 📁 Structure
```
docs/
├── Makefile                   # Build orchestration with parallel processing
├── source/conf.py            # Sphinx configuration with custom extensions  
├── rst_examples.py           # Dynamic example RST generation
├── dalapi/                   # Custom Sphinx extension for C++ API
│   ├── extension.py          # Main extension with path resolution
│   ├── directives.py         # Custom RST directives
│   ├── doxypy/              # Doxygen XML parser
│   └── generator.py         # RST content generation
├── doxygen/oneapi/          # Doxygen configuration
│   └── Doxyfile             # Points to cpp/oneapi/dal sources
└── source/                  # RST source files organized by API
```

## 🔧 Build System Integration

### Makefile Orchestration
```makefile
# Parallel processing optimization
SPHINXOPTS = "-j`nproc`"  # Uses all available CPU cores

# Multi-stage build targets
create_rst_examples:     # Dynamic RST generation from examples
    python3 rst_examples.py

doxygen:                 # C++ API extraction
    cd doxygen/oneapi && doxygen

parse-doxygen: doxygen   # Custom XML→YAML conversion
    python -m dalapi.doxypy.cli doxygen/oneapi/doxygen/xml --compact > build/tree.yaml

html: create_rst_examples # Production build with Intel branding
    sphinx-build -t use_intelname -b html source build

html-github: create_rst_examples # GitHub-optimized build
    sphinx-build -b html source build
```

### Sphinx Configuration Customizations
```python
# Custom extension stack
extensions = [
    'sphinx_prompt',                    # Code prompt styling
    'sphinx_substitution_extensions',   # Variable substitutions
    'sphinx.ext.extlinks',             # External link shortcuts
    'sphinx_tabs.tabs',                # Tabbed content
    'dalapi',                          # CUSTOM: oneDAL API integration
    'sphinx.ext.githubpages',          # GitHub Pages optimization
    'notfound.extension'               # 404 page handling
]

# Variable substitution system
substitutions = [
    ('|short_name|', 'oneDAL'),
    ('|daal_in_code|', 'daal')
]

# External link management
extlinks = {
    'cpp_example': ('https://github.com/uxlfoundation/oneDAL/tree/main/examples/daal/cpp/source/%s', None),
    'daal4py_example': ('https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/%s', None),
}
```

## 🎭 Advanced Customizations

### Custom dalapi Extension
```python
# Path resolution for complex build relationships
class PathResolver(object):
    def __init__(self, app, relative_doxyfile_dir, relative_sources_dir):
        self.base_dir = app.confdir
        self.doxyfile_dir = self.absjoin(self.base_dir, relative_doxyfile_dir)
        self.doxygen_xml = self.absjoin(self.doxyfile_dir, 'doxygen', 'xml')

# Processing pipeline: C++ Headers → Doxygen XML → dalapi Parser → RST Content → Sphinx HTML
```

### Dynamic Example Integration  
```python
# rst_examples.py: Auto-generates RST from example source code
def create_rst(filedir, filename, lang):
    rst_content = '.. _{}_{}:\n\n{}\n{}\n\n'.format(lang, filename, filename, '#' * len(filename))
    rst_content += '.. literalinclude:: ../../../../examples/oneapi/{}/source/{}/{}\n'.format(lang, filedir, filename)
    rst_content += '  :language: cpp\n'
    return rst_content
```

### Interface-Specific Processing
```bash
# Doxygen configuration targets oneAPI interface ONLY
INPUT = ../../../cpp/oneapi/dal

# But examples include both DAAL and oneAPI patterns
examples/daal/cpp/source/     # Traditional DAAL examples
examples/oneapi/cpp/source/   # Modern oneAPI examples  
examples/oneapi/dpc/source/   # GPU-accelerated examples
```

## 🎯 Documentation Generation Options

### Build Targets Available
```makefile
# Primary targets
make html              # Production build with Intel branding (-t use_intelname)
make html-github       # GitHub-optimized build (no Intel branding)
make html-test         # Test/validation build

# Component targets  
make create_rst_examples   # Generate example RST files
make doxygen              # Generate Doxygen XML from C++ headers
make parse-doxygen        # Convert Doxygen XML to structured YAML

# Parallel processing (automatic)
SPHINXOPTS = "-j`nproc`"  # Uses all available CPU cores
```

### Configuration Variants
```python
# Intel branding control via Sphinx tags
html: -t use_intelname     # Production: Intel-specific branding enabled
html-github:               # GitHub: Open-source branding

# Output customization
exclude_patterns = [       # Selectively exclude content
    'daal/data-management/numeric-tables/*.rst',
    'daal/algorithms/dbscan/distributed-steps/*',
    'onedal/algorithms/.*/includes/*'
]
```

### Interface Targeting
- **API Documentation**: oneAPI interface ONLY via Doxygen (`INPUT = ../../../cpp/oneapi/dal`)
- **Example Integration**: Both DAAL and oneAPI examples via `rst_examples.py`
- **Cross-references**: Automatic linking between examples and API docs

## 🏛️ Customization Architecture

### Extension System
- **dalapi**: Custom Sphinx extension bridges Doxygen→Sphinx
- **Path Resolution**: Complex build relationship management
- **Custom Directives**: oneDAL-specific RST directives
- **XML Processing**: Advanced Doxygen XML parsing and transformation

### Content Generation
- **Dynamic RST**: Examples auto-converted from source code
- **API Integration**: C++ headers → XML → YAML → RST → HTML pipeline
- **Variable Substitutions**: Global replacements (`|short_name|` → `oneDAL`)
- **External Links**: Automated GitHub link generation

### Multi-Format Support
- **Tabbed Content**: `sphinx_tabs.tabs` for interface comparisons
- **Code Prompts**: `sphinx_prompt` for terminal examples  
- **Modern Theme**: `sphinx_book_theme` with responsive design
- **GitHub Integration**: Optimized for GitHub Pages deployment

## 🎯 Critical Generation Rules

### Build Dependencies
- **Sequential Processing**: RST generation → Doxygen → XML parsing → Sphinx build
- **Parallel Optimization**: Use all CPU cores for Sphinx processing
- **Interface Separation**: oneAPI gets full API docs, DAAL examples included
- **Example Synchronization**: Auto-generated RST stays in sync with source code

### Customization Guidelines  
- **Extension Configuration**: Modify `dalapi` extension for API processing changes
- **Build Targets**: Use appropriate target for deployment context (Intel vs GitHub)
- **Content Exclusion**: Update `exclude_patterns` for content filtering
- **Link Management**: Maintain `extlinks` for external references

## 📖 Further Reading
- **[cpp/AGENTS.md](../cpp/AGENTS.md)** - C++ implementation context
- **[cpp/oneapi/AGENTS.md](../cpp/oneapi/AGENTS.md)** - oneAPI interface patterns
- **[examples/AGENTS.md](../examples/AGENTS.md)** - Example integration patterns  
- **[dev/AGENTS.md](../dev/AGENTS.md)** - Build system architecture
