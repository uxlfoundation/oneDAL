
# Deployment and Distribution - AI Agents Context

> **Purpose**: Context for AI agents working with oneDAL deployment, packaging, and distribution processes.

## 🎯 Deployment Overview

**oneDAL** deployment involves multiple packaging formats and distribution channels:
- **Multi-Format**: Multiple package formats for different use cases
- **Cross-Platform**: Support for Linux, Windows, macOS
- **Integration**: Works with scikit-learn-intelex for Python distribution
- **Enterprise**: Commercial distribution through Intel channels
- **Open Source**: Community distribution through UXL Foundation

## 🏗️ Structure
```
deploy/
├── local/      # Local development deployment
├── nuget/      # .NET package distribution
└── pkg-config/ # pkg-config distribution
```

## 🔧 Deployment Formats

### Local Development Deployment
- **Purpose**: Development and testing environments
- **Format**: Binary + configuration files
- **Platform**: All supported platforms

### NuGet Package Distribution
- **Purpose**: .NET ecosystem integration
- **Target**: Windows developers using .NET
- **Format**: .nupkg files

### pkg-config Distribution
- **Purpose**: Unix/Linux system integration
- **Format**: .pc files
- **Platform**: Linux, macOS, Unix systems

## 📦 Package Creation

### NuGet Package
```bash
./deploy/nuget/prepare_dal_nuget.sh
# Includes: Native libraries, headers, documentation, license
```

### pkg-config Files
```bash
python deploy/pkg-config/generate_pkgconfig.py
# Outputs: onedal.pc, onedal_c.pc, platform-specific configs
```

### Local Development Setup
```bash
# Linux/macOS
source deploy/local/vars_lnx.sh

# Windows
deploy/local/vars_win.bat
# Sets: ONEDAL_ROOT, PATH updates, Library paths
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
- **Open Source**: GitHub releases, package managers, documentation
- **Commercial**: Intel channels, enterprise support, performance tuning
- **Community**: UXL Foundation, community contributions, standards compliance

## 📋 Deployment Checklist

### Pre-Release Validation
- All tests pass (Make, CMake, Bazel)
- Performance benchmarks meet targets
- Documentation complete and accurate
- Cross-platform compatibility verified

### Package Creation
- NuGet package builds successfully
- pkg-config files generated correctly
- Local deployment scripts work on all platforms

### Distribution
- Release notes comprehensive
- Binary compatibility maintained
- Integration with scikit-learn-intelex verified

## 🔍 Common Deployment Scenarios

### New Release Deployment
1. Build validation → Package creation → Testing → Distribution → Documentation

### Hotfix Deployment
1. Issue identification → Fix development → Package update → Rapid distribution

### Platform Addition
1. Platform support → Build configuration → Testing → Package updates → Documentation

## 🛠️ Troubleshooting

### Common Issues
- **Package Installation**: Check platform compatibility and dependencies
- **Build Failures**: Verify build system configuration and C++17 compliance
- **Integration Issues**: Check scikit-learn-intelex compatibility
- **Performance**: Validate optimization flags and compiler settings

### Debugging Steps
1. Environment check → Dependency validation → Platform verification → Integration testing

## 📖 Further Reading
- **[AGENTS.md](../AGENTS.md)** - Main repository context
- **[dev/AGENTS.md](../dev/AGENTS.md)** - Development tools context
- **[ci/AGENTS.md](../ci/AGENTS.md)** - CI infrastructure context
