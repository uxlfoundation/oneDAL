# Copyright contributors to the oneDAL project
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

# GitHub Copilot Instructions for oneDAL

## 🎯 **PRIMARY GOAL: PR Review Assistance**

**GitHub Copilot's main purpose in this repository is to assist with PR reviews and validation.**

This directory contains custom instructions for GitHub Copilot to understand the oneDAL repository structure, coding standards, and development patterns.

## 📁 Instruction Files Structure

- **[general.md](general.md)** - General repository context and rules
- **[cpp.md](cpp.md)** - C++ development guidelines and patterns
- **[coding-guidelines.md](coding-guidelines.md)** - Comprehensive coding standards and best practices
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

## 📋 Key Rules Summary

1. **🔴 Build System Priority**: Make is PRIMARY for production, CMake for end-users, Bazel for development
2. **Interface Selection**: Use oneAPI for new code, DAAL for legacy maintenance
3. **C++ Standards**: C++17 maximum, C++14 minimum (no C++20/23 for compatibility)
4. **Code Quality**: Modern C++ patterns, RAII, smart pointers
5. **Context Awareness**: Always check file path for appropriate patterns
6. **C++17 Compliance**: Strict adherence to C++17 maximum standard
7. **Integration**: Works with scikit-learn-intelex project
8. **🎯 PR Review Focus**: Primary goal is PR review assistance
9. **Coding Standards**: Follow comprehensive coding guidelines for consistency

## 🔍 **PR Review Assistance (PRIMARY FOCUS)**

### Common PR Review Scenarios
- **New Algorithm Implementation**: Check interface consistency and Make compatibility
- **Build System Changes**: Verify Make compatibility FIRST (🔴 CRITICAL)
- **Test Additions**: Validate Bazel test configuration for development
- **Documentation Updates**: Ensure accuracy and completeness
- **Performance Changes**: Verify Make build performance impact
- **Integration Changes**: Check scikit-learn-intelex compatibility
- **Code Quality**: Verify coding standards and best practices


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
- **[Build Systems](build-systems.md)** - Build system guidance
- **[C++ Development](cpp.md)** - C++ coding guidelines
- **[Coding Guidelines](coding-guidelines.md)** - Comprehensive coding standards
- **[Examples](examples.md)** - Code pattern examples
- **[Documentation](documentation.md)** - Documentation standards
- **[CI Workflows](ci-workflows.md)** - CI/CD validation guidance


