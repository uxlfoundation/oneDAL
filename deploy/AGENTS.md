# Deployment and Distribution - AI Agents Context

> **Purpose**: This file provides comprehensive context for AI agents working with oneDAL deployment, packaging, and distribution processes.

## 🎯 Deployment Overview

**oneDAL** deployment involves multiple packaging formats and distribution channels to support different user needs and platforms.

### Key Deployment Characteristics
- **Multi-Format**: Multiple package formats for different use cases
- **Cross-Platform**: Support for Linux, Windows, macOS
- **Integration**: Works with scikit-learn-intelex for Python distribution
- **Enterprise**: Commercial distribution through Intel channels
- **Open Source**: Community distribution through UXL Foundation

## 🏗️ Deployment Structure

```
deploy/
├── local/                    # Local development deployment
│   ├── config.txt           # Local configuration
│   ├── dal                  # Local binary
│   ├── vars_win.bat         # Windows environment variables
│   └── vars_lnx.sh          # Linux environment variables
├── nuget/                    # .NET package distribution
│   ├── inteldal.nuspec.tpl  # NuGet package template
│   └── prepare_dal_nuget.sh # NuGet preparation script
└── pkg-config/              # pkg-config distribution
    ├── generate_pkgconfig.py # pkg-config generation script
    └── pkg-config.tpl       # pkg-config template
```

## 🔧 Deployment Processes

### Local Development Deployment
- **Purpose**: Development and testing environments
- **Target**: Local machine setup
- **Format**: Binary + configuration files
- **Platform**: All supported platforms

### NuGet Package Distribution
- **Purpose**: .NET ecosystem integration
- **Target**: Windows developers using .NET
- **Format**: .nupkg files
- **Platform**: Windows (primary), cross-platform support

### pkg-config Distribution
- **Purpose**: Unix/Linux system integration
- **Target**: System package managers and build tools
- **Format**: .pc files
- **Platform**: Linux, macOS, Unix systems

## 📦 Package Creation Workflows

### NuGet Package Creation
```bash
# Generate NuGet package
./deploy/nuget/prepare_dal_nuget.sh

# Package includes:
# - Native libraries (x86, x64, ARM64)
# - Header files
# - Documentation
# - License and metadata
```

### pkg-config Generation
```python
# Generate pkg-config files
python deploy/pkg-config/generate_pkgconfig.py

# Outputs:
# - onedal.pc
# - onedal_c.pc
# - Platform-specific configurations
```

### Local Development Setup
```bash
# Linux/macOS
source deploy/local/vars_lnx.sh

# Windows
deploy/local/vars_win.bat

# Sets environment variables:
# - ONEDAL_ROOT
# - PATH updates
# - Library paths
```

## 🔄 Integration with scikit-learn-intelex

### Cross-Repository Considerations
- **Shared Validation**: Common test suites and validation procedures
- **Version Alignment**: Ensure compatibility between oneDAL and scikit-learn-intelex
- **Distribution Coordination**: Coordinate releases for optimal user experience

### Validation Aspects
- **API Compatibility**: Verify oneDAL API changes don't break scikit-learn-intelex
- **Performance Testing**: Ensure performance improvements are maintained
- **Platform Coverage**: Test on all supported platforms

## 🚀 Distribution Strategies

### Open Source Distribution
- **GitHub Releases**: Source code and pre-built binaries
- **Package Managers**: Integration with system package managers
- **Documentation**: Comprehensive user and developer guides

### Commercial Distribution
- **Intel Channels**: Enterprise support and distribution
- **Performance Tuning**: Optimized builds for Intel platforms
- **Support**: Professional support and maintenance

### Community Distribution
- **UXL Foundation**: Open source community distribution
- **Contributions**: Community-driven improvements
- **Standards**: Industry standard compliance

## 📋 Deployment Checklist

### Pre-Release Validation
- [ ] All tests pass (Make, CMake, Bazel)
- [ ] Performance benchmarks meet targets
- [ ] Documentation is complete and accurate
- [ ] License and attribution files are correct
- [ ] Cross-platform compatibility verified

### Package Creation
- [ ] NuGet package builds successfully
- [ ] pkg-config files are generated correctly
- [ ] Local deployment scripts work on all platforms
- [ ] Package metadata is accurate and complete

### Distribution
- [ ] Release notes are comprehensive
- [ ] Binary compatibility is maintained
- [ ] Integration with scikit-learn-intelex is verified
- [ ] Platform-specific issues are addressed

## 🔍 Common Deployment Scenarios

### New Release Deployment
1. **Build Validation**: Ensure all build systems work
2. **Package Creation**: Generate all package formats
3. **Testing**: Validate packages on target platforms
4. **Distribution**: Deploy to all channels
5. **Documentation**: Update user guides and examples

### Hotfix Deployment
1. **Issue Identification**: Identify critical issue
2. **Fix Development**: Develop and test fix
3. **Package Update**: Update affected packages
4. **Rapid Distribution**: Deploy to critical users
5. **Full Release**: Include in next regular release

### Platform Addition
1. **Platform Support**: Add new platform support
2. **Build Configuration**: Configure build systems
3. **Testing**: Validate on new platform
4. **Package Updates**: Update all package formats
5. **Documentation**: Update platform-specific guides

## 🛠️ Troubleshooting

### Common Issues
- **Package Installation Failures**: Check platform compatibility and dependencies
- **Build Failures**: Verify build system configuration and C++17 compliance
- **Integration Issues**: Check scikit-learn-intelex compatibility
- **Performance Degradation**: Validate optimization flags and compiler settings

### Debugging Steps
1. **Environment Check**: Verify deployment environment variables
2. **Dependency Validation**: Check all required dependencies
3. **Platform Verification**: Ensure platform-specific configurations
4. **Integration Testing**: Test with scikit-learn-intelex

## 📚 Related Context Files

### For This Area
- **[AGENTS.md](../../AGENTS.md)** - Main repository context
- **[dev/AGENTS.md](../../dev/AGENTS.md)** - Development tools context

### For Other Areas
- **[Build Systems](../../.github/instructions/build-systems.md)** - Build configuration guidance
- **[Examples](../../.github/instructions/examples.md)** - Usage examples
- **[Documentation](../../.github/instructions/documentation.md)** - Documentation standards

## 🔄 Cross-Reference Navigation

**For Deployment Tasks**: Use this file for deployment-specific guidance
**For Build Issues**: Refer to build system context files
**For Integration**: Check scikit-learn-intelex compatibility
**For User Support**: Use examples and documentation context

---

**Note**: This file provides deployment context for AI agents. For detailed implementation, refer to the specific deployment scripts and configuration files in this directory.
