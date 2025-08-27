# GitHub Copilot Instructions for oneDAL

## 🎯 **PRIMARY GOAL: PR Review Assistance**

**GitHub Copilot's main purpose in this repository is to assist with PR reviews and validation.**

This directory contains custom instructions for GitHub Copilot to understand the oneDAL repository structure, coding standards, and development patterns.

## 📁 Instruction Files Structure

- **[general.md](general.md)** - General repository context and rules
- **[cpp.md](cpp.md)** - C++ development guidelines and patterns
- **[cpp17-constraints.md](cpp17-constraints.md)** - C++17 constraints and best practices
- **[build-systems.md](build-systems.md)** - Build system configurations and rules
- **[documentation.md](documentation.md)** - Documentation standards and patterns
- **[examples.md](examples.md)** - Example code guidelines and patterns
- **[ci-workflows.md](ci-workflows.md)** - CI/CD workflows and validation

## 🎯 How to Use

These instruction files are automatically loaded by GitHub Copilot when working in the oneDAL repository. They provide context-aware suggestions based on:

- **File location** in the repository
- **File type** and content
- **Development context** (new code vs. legacy maintenance)
- **PR Review scenarios** (primary focus)

## 🔴 **CRITICAL BUILD SYSTEM PRIORITY FOR PR REVIEWS**

### Build System Priority Order (CRITICAL)
1. **🔴 Make (CRITICAL)**: Primary build system for production builds
2. **🟡 CMake (IMPORTANT)**: End-user integration support (find_package)
3. **🟢 Bazel (DEVELOPMENT)**: Development and testing (ongoing migration)

**🚨 WHY MAKE FIRST?** Make is the production build system used for releases. All changes MUST work with Make builds for production deployment.

## 📋 Key Rules Summary

1. **🔴 Build System Priority**: Make is PRIMARY for production, CMake for end-users, Bazel for development
2. **Interface Selection**: Use oneAPI for new code, DAAL for legacy maintenance
3. **C++ Standards**: C++17 maximum, C++14 minimum (no C++20/23 for compatibility)
4. **Code Quality**: Modern C++ patterns, RAII, smart pointers
5. **Context Awareness**: Always check file path for appropriate patterns
6. **C++17 Compliance**: Strict adherence to C++17 maximum standard
7. **Integration**: Works with scikit-learn-intelex project
8. **🎯 PR Review Focus**: Primary goal is PR review assistance

## 🔍 **PR Review Assistance (PRIMARY FOCUS)**

### Common PR Review Scenarios
- **New Algorithm Implementation**: Check interface consistency and Make compatibility
- **Build System Changes**: Verify Make compatibility FIRST (🔴 CRITICAL)
- **Test Additions**: Validate Bazel test configuration for development
- **Documentation Updates**: Ensure accuracy and completeness
- **Performance Changes**: Verify Make build performance impact
- **Integration Changes**: Check scikit-learn-intelex compatibility

### PR Review Checklist Template
```markdown
## PR Review Checklist

### 🔴 Build System Validation (CRITICAL)
- [ ] **Make build succeeds** (production validation)
- [ ] **CMake integration works** (end-user support)
- [ ] **Bazel tests pass** (development validation)

### 🟡 Code Quality
- [ ] **C++17 compliance maintained** (no C++20/23)
- [ ] **Interface consistency preserved** (DAAL vs oneAPI)
- [ ] **Error handling implemented** (proper exception safety)
- [ ] **Documentation updated** (accurate and complete)

### 🟡 Cross-Repository Impact
- [ ] **scikit-learn-intelex compatibility** assessed
- [ ] **API changes documented** for integration
- [ ] **Performance impact** evaluated
- [ ] **Breaking changes** clearly identified

### 🟢 Development Workflow
- [ ] **Examples build and run** correctly
- [ ] **Tests provide adequate coverage**
- [ ] **Code follows project patterns**
- [ ] **No platform-specific hardcoding**
```

## 🔗 Related Documentation

- **[AGENTS.md](../../AGENTS.md)** - Comprehensive AI agent context
- **[CONTRIBUTING.md](../../CONTRIBUTING.md)** - Contribution guidelines
- **[cpp/AGENTS.md](../../cpp/AGENTS.md)** - C++ implementation context

## 🔄 Cross-References

### For AI Agents
- **[cpp/daal/AGENTS.md](../../cpp/daal/AGENTS.md)** - Traditional DAAL interface context
- **[cpp/oneapi/AGENTS.md](../../cpp/oneapi/AGENTS.md)** - Modern oneAPI interface context
- **[dev/AGENTS.md](../../dev/AGENTS.md)** - Development tools context
- **[dev/bazel/AGENTS.md](../../dev/bazel/AGENTS.md)** - Bazel build system context (🟢 development only)
- **[dev/make/AGENTS.md](../../dev/make/AGENTS.md)** - Make build system context (🔴 PRIMARY for production)
- **[docs/AGENTS.md](../../docs/AGENTS.md)** - Documentation context
- **[examples/AGENTS.md](../../examples/AGENTS.md)** - Examples context
- **[samples/AGENTS.md](../../samples/AGENTS.md)** - Samples context
- **[deploy/AGENTS.md](../../deploy/AGENTS.md)** - Deployment and distribution context
- **[ci/AGENTS.md](../../ci/AGENTS.md)** - CI infrastructure and workflows context

### For Copilot Instructions
- **[Build Systems](../../.github/instructions/build-systems.md)** - Build system guidance
- **[C++ Development](../../.github/instructions/cpp.md)** - C++ coding guidelines
- **[Examples](../../.github/instructions/examples.md)** - Code pattern examples
- **[Documentation](../../.github/instructions/documentation.md)** - Documentation standards
- **[CI Workflows](../../.github/instructions/ci-workflows.md)** - CI/CD validation guidance

## 🚨 **Critical Reminders for PR Review**

1. **🔴 Make compatibility is CRITICAL** - verify FIRST
2. **🟡 CMake integration is IMPORTANT** - verify SECOND  
3. **🟢 Bazel testing is DEVELOPMENT** - verify THIRD
4. **Production builds use Make** - not Bazel
5. **End-users use CMake** - ensure find_package works
6. **Development uses Bazel** - for testing and CI/CD
7. **C++17 maximum standard** - no C++20/23 features
8. **Interface consistency** - don't mix DAAL and oneAPI
9. **Cross-repository impact** - consider scikit-learn-intelex

## 📊 **Quick Decision Guide**

### New Algorithm Implementation?
- **Interface**: Use oneAPI (`cpp/oneapi/`)
- **Build**: Ensure Make compatibility (🔴 CRITICAL)
- **Testing**: Include Bazel tests (🟢 development)
- **Documentation**: Update relevant docs

### Legacy Code Maintenance?
- **Interface**: Use DAAL (`cpp/daal/`)
- **Build**: Maintain Make compatibility (🔴 CRITICAL)
- **Compatibility**: Preserve backward compatibility
- **Testing**: Ensure existing tests pass

### Build System Changes?
- **Priority**: Make compatibility FIRST (🔴 CRITICAL)
- **Integration**: CMake support SECOND (🟡 IMPORTANT)
- **Development**: Bazel workflow THIRD (🟢 DEVELOPMENT)
- **Validation**: Test all build systems

---

**Note**: These instructions are automatically applied by GitHub Copilot. For detailed context, refer to the main AGENTS.md files in the repository.

**🎯 PRIMARY GOAL**: Assist with PR reviews by providing accurate, context-aware suggestions that prioritize production build compatibility and code quality.

**🚨 CRITICAL**: For PR reviews, always verify Make compatibility FIRST - this is the production build system!
