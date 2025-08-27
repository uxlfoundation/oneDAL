# PR Review Templates for GitHub Copilot

## 🎯 **PRIMARY PURPOSE: PR Review Assistance**

This file provides templates and quick reference guides for GitHub Copilot to assist with PR reviews in the oneDAL repository.

## 🔴 **CRITICAL BUILD SYSTEM PRIORITY**

### Build System Validation Order (CRITICAL FOR PR REVIEWS)
1. **🔴 Make (CRITICAL)**: Primary build system for production builds
2. **🟡 CMake (IMPORTANT)**: End-user integration support (find_package)
3. **🟢 Bazel (DEVELOPMENT)**: Development and testing (ongoing migration)

**🚨 WHY MAKE FIRST?** Make is the production build system used for releases. All changes MUST work with Make builds for production deployment.

## 📋 **PR Review Checklist Templates**

### 🔴 **Standard PR Review Checklist**

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

### 🎯 **Algorithm Implementation Review**

```markdown
## Algorithm Implementation Review

### Interface Selection
- [ ] **New algorithm**: Uses oneAPI interface (`cpp/oneapi/`)
- [ ] **Legacy update**: Uses DAAL interface (`cpp/daal/`)
- [ ] **No mixing**: DAAL and oneAPI not mixed in same file

### Build Compatibility
- [ ] **Make compatibility** (🔴 CRITICAL for production)
- [ ] **CMake integration** (🟡 IMPORTANT for end-users)
- [ ] **Bazel testing** (🟢 DEVELOPMENT workflow)

### Code Quality
- [ ] **C++17 compliance** (no C++20/23 features)
- [ ] **Modern patterns** (smart pointers, RAII)
- [ ] **Error handling** (proper exception safety)
- [ ] **Performance** (no regression introduced)

### Testing and Validation
- [ ] **Unit tests** included and passing
- [ ] **Integration tests** updated if needed
- [ ] **Performance tests** validate improvements
- [ ] **Examples** demonstrate usage
```

### 🏗️ **Build System Changes Review**

```markdown
## Build System Changes Review

### 🔴 Make Build System (CRITICAL)
- [ ] **Production builds** still work with Make
- [ ] **Performance characteristics** maintained
- [ ] **Dependencies resolved** properly
- [ ] **No breaking changes** introduced

### 🟡 CMake Integration (IMPORTANT)
- [ ] **find_package(oneDAL)** still works
- [ ] **End-user projects** can build successfully
- [ ] **Cross-platform support** maintained
- [ ] **IDE integration** works properly

### 🟢 Bazel Development (DEVELOPMENT)
- [ ] **Development workflow** preserved
- [ ] **Test suites** run successfully
- [ ] **CI/CD pipelines** work correctly
- [ ] **Dependency management** functional

### Validation
- [ ] **All build systems** tested
- [ ] **Cross-platform** compatibility verified
- [ ] **Performance impact** assessed
- [ ] **Documentation** updated
```

### 📚 **Documentation Updates Review**

```markdown
## Documentation Updates Review

### Content Quality
- [ ] **Technical accuracy** maintained
- [ ] **Completeness** of information
- [ ] **Clarity** of explanations
- [ ] **Examples** are accurate and runnable

### Build System Documentation
- [ ] **Make priority** clearly stated (🔴 CRITICAL)
- [ ] **CMake integration** documented (🟡 IMPORTANT)
- [ ] **Bazel usage** clarified (🟢 DEVELOPMENT)
- [ ] **Build system hierarchy** explained

### Cross-References
- [ ] **Internal links** work correctly
- [ ] **External references** are valid
- [ ] **Related documentation** linked
- [ ] **Navigation** is clear

### Examples and Code
- [ ] **Code examples** compile and run
- [ ] **Build instructions** are accurate
- [ ] **Platform differences** documented
- [ ] **Troubleshooting** guidance included
```

### 🔄 **Cross-Repository Integration Review**

```markdown
## Cross-Repository Integration Review

### scikit-learn-intelex Compatibility
- [ ] **API changes** don't break integration
- [ ] **Performance improvements** maintained
- [ ] **Feature parity** preserved
- [ ] **Error handling** consistent

### Integration Documentation
- [ ] **Changes documented** for integration
- [ ] **Migration guide** provided if needed
- [ ] **Breaking changes** clearly identified
- [ ] **Performance impact** documented

### Testing and Validation
- [ ] **Integration tests** pass
- [ ] **Cross-repository** validation successful
- [ ] **Performance benchmarks** updated
- [ ] **Compatibility matrix** current
```

## 🚀 **Quick Decision Guides**

### New Feature Implementation?
```
Interface: oneAPI (cpp/oneapi/)
Build Priority: Make compatibility FIRST (🔴 CRITICAL)
Testing: Bazel tests (🟢 development)
Documentation: Update relevant docs
Integration: Consider scikit-learn-intelex impact
```

### Legacy Code Maintenance?
```
Interface: DAAL (cpp/daal/)
Build Priority: Make compatibility FIRST (🔴 CRITICAL)
Compatibility: Preserve backward compatibility
Testing: Ensure existing tests pass
Documentation: Update if needed
```

### Build System Changes?
```
Priority: Make compatibility FIRST (🔴 CRITICAL)
Integration: CMake support SECOND (🟡 IMPORTANT)
Development: Bazel workflow THIRD (🟢 DEVELOPMENT)
Validation: Test all build systems
Documentation: Update build instructions
```

### Performance Changes?
```
Make Build: Performance maintained or improved
Runtime: Algorithm performance preserved
Memory: Memory efficiency maintained
Platforms: Works across supported platforms
Documentation: Performance impact documented
```

## 🔍 **Common PR Review Scenarios**

### 1. **New Algorithm Implementation**
- **Check**: Interface consistency (oneAPI vs DAAL)
- **Verify**: Make compatibility (🔴 CRITICAL)
- **Validate**: CMake integration (🟡 IMPORTANT)
- **Test**: Bazel tests (🟢 DEVELOPMENT)
- **Document**: Usage examples and API docs

### 2. **Build System Changes**
- **Priority**: Make compatibility FIRST (🔴 CRITICAL)
- **Secondary**: CMake integration (🟡 IMPORTANT)
- **Tertiary**: Bazel workflow (🟢 DEVELOPMENT)
- **Validation**: Test all build systems
- **Documentation**: Update build instructions

### 3. **Performance Improvements**
- **Production**: Make build performance maintained
- **Runtime**: Algorithm performance improved
- **Memory**: Memory efficiency maintained
- **Platforms**: Cross-platform compatibility
- **Documentation**: Performance impact documented

### 4. **Bug Fixes**
- **Root Cause**: Issue properly identified
- **Fix**: Solution addresses root cause
- **Testing**: Tests cover the fix
- **Regression**: No new issues introduced
- **Documentation**: Fix documented if needed

### 5. **Documentation Updates**
- **Accuracy**: Information is technically correct
- **Completeness**: All changes documented
- **Examples**: Code examples compile and run
- **Cross-References**: Links work correctly
- **Build System**: Priority order clearly stated

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

## 📊 **Review Priority Matrix**

| Change Type | Make Priority | CMake Priority | Bazel Priority |
|-------------|---------------|----------------|----------------|
| **Production Bug Fix** | 🔴 CRITICAL | 🟡 IMPORTANT | 🟢 DEVELOPMENT |
| **New Feature** | 🔴 CRITICAL | 🟡 IMPORTANT | 🟢 DEVELOPMENT |
| **Performance** | 🔴 CRITICAL | 🟡 IMPORTANT | 🟢 DEVELOPMENT |
| **Documentation** | 🟡 IMPORTANT | 🟡 IMPORTANT | 🟢 DEVELOPMENT |
| **Build System** | 🔴 CRITICAL | 🔴 CRITICAL | 🔴 CRITICAL |
| **Testing** | 🟡 IMPORTANT | 🟡 IMPORTANT | 🔴 CRITICAL |

## 🔄 **Cross-Reference Navigation**

### For This Area
- **[AGENTS.md](../../AGENTS.md)** - Main repository context
- **[general.md](general.md)** - General repository rules

### For Other Areas
- **[Build Systems](../../.github/instructions/build-systems.md)** - Build system guidance
- **[C++ Development](../../.github/instructions/cpp.md)** - C++ coding guidelines
- **[Examples](../../.github/instructions/examples.md)** - Code pattern examples
- **[CI Workflows](../../.github/instructions/ci-workflows.md)** - CI/CD validation

---

**Note**: Use these templates to ensure comprehensive and consistent PR reviews.

**🎯 PRIMARY GOAL**: Assist with PR reviews by providing structured, comprehensive validation that prioritizes production build compatibility.

**🚨 CRITICAL**: For PR reviews, always verify Make compatibility FIRST - this is the production build system!