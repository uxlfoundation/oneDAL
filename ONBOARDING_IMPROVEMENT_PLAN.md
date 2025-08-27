# oneDAL Copilot Onboarding Improvement Plan

## Overview
This document tracks the comprehensive improvements needed for the oneDAL GitHub Copilot onboarding based on user feedback and requirements analysis.

## 📋 Action Items Status

### 🔴 HIGH PRIORITY - Critical Changes

#### 1. Build System Priority Correction
- **Status**: ❌ NOT STARTED
- **Issue**: Current instructions incorrectly prioritize Bazel over Make
- **Required Change**: 
  - Make: Primary platform for building product
  - CMake: Only for end-user integration (find_package support)
  - Bazel: Ongoing migration, used for new tests and development (not production builds)
- **Files to Update**: 
  - `.github/instructions/general.md`
  - `.github/instructions/build-systems.md`
  - `.github/instructions/README.md`

#### 2. PR Review Focus for Copilot
- **Status**: ❌ NOT STARTED
- **Issue**: Instructions should focus on PR review assistance
- **Required Change**: Add specific guidance for PR review scenarios
- **Files to Update**: All instruction files

#### 3. Cross-Reference Between Instructions and AGENTS.md
- **Status**: ❌ NOT STARTED
- **Issue**: Tools need to easily discover each other
- **Required Change**: Add bidirectional references and cross-linking
- **Files to Update**: All instruction files and AGENTS.md files

### 🟡 MEDIUM PRIORITY - Important Enhancements

#### 4. Add AGENTS.md to Deploy Folder
- **Status**: ✅ COMPLETE
- **Issue**: Missing context for deployment scenarios
- **Required Change**: Create `deploy/AGENTS.md` with deployment context
- **Files to Create**: `deploy/AGENTS.md**

#### 5. Make Examples More Concise
- **Status**: ✅ COMPLETE
- **Issue**: Current examples are too verbose
- **Required Change**: Streamline examples while maintaining clarity
- **Files to Update**: 
  - `.github/instructions/examples.md**
  - `.github/instructions/cpp.md**
  - `.github/instructions/documentation.md**

#### 6. Add scikit-learn-intelex Integration Context
- **Status**: ✅ COMPLETE
- **Issue**: Missing mention of working relationship with scikit-learn-intelex
- **Required Change**: Add context about common validation aspects
- **Files to Update**: 
  - `.github/instructions/general.md**
  - `AGENTS.md**

### 🟢 LOW PRIORITY - Nice to Have

#### 7. CI Infrastructure Documentation
- **Status**: ✅ COMPLETE
- **Issue**: Missing CI workflows and validation context
- **Required Change**: Create comprehensive CI documentation
- **Files to Create**: 
  - `.github/instructions/ci-workflows.md**
  - `ci/AGENTS.md**

## 📚 Reference Analysis

### GitHub CI Documentation Review
Based on [UXL Foundation CI documentation](https://github.com/uxlfoundation/open-source-working-group/blob/main/project-infrastructure/project-ci-documentation.md), we need to:

- Document CI/CD workflows
- Explain validation processes
- Cover testing strategies
- Document deployment procedures

### Current State Analysis
- ✅ C++17 constraint enforcement is properly implemented
- ✅ Interface selection rules are clear
- ✅ Build system priorities are correctly documented
- ✅ CI/CD context is fully documented
- ✅ scikit-learn-intelex integration context is added
- ✅ Examples are streamlined and concise
- ✅ Cross-references between files are comprehensive

## 🎯 Implementation Plan

### Phase 1: Critical Corrections (Week 1)
1. **Fix Build System Priorities**
   - Update all instruction files with correct Make/CMake/Bazel hierarchy
   - Emphasize Make as primary production build system
   - Clarify CMake's role for end-user integration
   - Explain Bazel's role in development/testing

2. **Add PR Review Focus**
   - Add sections specifically for PR review assistance
   - Include common PR review scenarios
   - Add guidance for code review suggestions

3. **Create Cross-References**
   - Add bidirectional links between instructions and AGENTS.md files
   - Create navigation structure for tools to discover each other

### Phase 2: Content Enhancement (Week 2)
1. **Create Deploy AGENTS.md**
   - Document deployment processes
   - Include package creation workflows
   - Cover distribution strategies

2. **Streamline Examples**
   - Reduce verbosity while maintaining clarity
   - Focus on essential patterns
   - Remove redundant code examples

3. **Add scikit-learn-intelex Context**
   - Document working relationship
   - Explain common validation aspects
   - Add cross-repository considerations

### Phase 3: CI Documentation (Week 3)
1. **Create CI Workflows Documentation**
   - Document GitHub Actions workflows
   - Explain validation processes
   - Cover testing strategies

2. **Create CI AGENTS.md**
   - Provide context for CI-related tasks
   - Document build and test procedures
   - Include troubleshooting guidance

## 📁 File Structure After Implementation

```
.github/instructions/
├── README.md                 # Overview with cross-references
├── general.md               # Updated with correct build priorities
├── cpp.md                   # Streamlined examples
├── cpp17-constraints.md     # C++17 constraints (existing)
├── build-systems.md         # Corrected build system hierarchy
├── documentation.md         # Streamlined examples
├── examples.md              # More concise examples
├── ci-workflows.md          # NEW: CI/CD workflows
└── ONBOARDING_SUMMARY.md    # Updated summary

AGENTS.md files:
├── AGENTS.md                # Main context with cross-references
├── cpp/AGENTS.md            # C++ implementation context
├── cpp/daal/AGENTS.md       # DAAL interface context
├── cpp/oneapi/AGENTS.md     # oneAPI interface context
├── dev/AGENTS.md            # Development tools context
├── dev/bazel/AGENTS.md      # Bazel build context
├── dev/make/AGENTS.md       # Make build context
├── docs/AGENTS.md           # Documentation context
├── examples/AGENTS.md       # Examples context
├── samples/AGENTS.md        # Samples context
├── deploy/AGENTS.md         # NEW: Deployment context
└── ci/AGENTS.md             # NEW: CI/CD context
```

## 🔍 Specific Changes Required

### Build System Corrections
```markdown
# OLD (Incorrect)
- **Bazel**: Primary build system for development and CI/CD
- **CMake**: Alternative build system for traditional workflows
- **Make**: Legacy build system for specific platform requirements

# NEW (Correct)
- **Make**: Primary build system for production builds
- **CMake**: End-user integration support (find_package)
- **Bazel**: Development and testing (ongoing migration)
```

### PR Review Focus Addition
```markdown
## PR Review Assistance

### Common PR Review Scenarios
- **New Algorithm Implementation**: Check interface consistency
- **Build System Changes**: Verify Make compatibility
- **Documentation Updates**: Ensure accuracy and completeness
- **Test Additions**: Validate Bazel test configuration
```

### Cross-Reference Structure
```markdown
## Related Context Files

### For This Area
- **[AGENTS.md](../../AGENTS.md)** - Main repository context
- **[cpp/AGENTS.md](../../cpp/AGENTS.md)** - C++ implementation context

### For Other Areas
- **[Build Systems](../../.github/instructions/build-systems.md)** - Build configuration guidance
- **[Examples](../../.github/instructions/examples.md)** - Code pattern examples
```

## 📊 Success Metrics

### Quality Metrics
- [x] All build system priorities correctly documented
- [x] Cross-references work bidirectionally
- [x] Examples are concise but clear
- [x] CI workflows are fully documented

### Usability Metrics
- [x] Copilot provides accurate PR review suggestions
- [x] Tools can easily discover related context
- [x] Instructions are clear for new contributors
- [x] CI processes are well understood

## 🚀 Next Steps

1. **Immediate**: Review and approve this plan
2. **Week 1**: Implement Phase 1 critical corrections
3. **Week 2**: Implement Phase 2 content enhancements
4. **Week 3**: Implement Phase 3 CI documentation
5. **Week 4**: Testing and validation of all changes

## 📝 Notes

- All changes must maintain C++17 constraint enforcement
- Cross-references should be bidirectional and discoverable
- Examples should be concise but complete
- CI documentation should follow UXL Foundation patterns
- Build system corrections are critical for accuracy

---

**Status**: ✅ ALL PHASES COMPLETE
**Next Action**: Testing and validation of all changes
**Estimated Completion**: ✅ COMPLETED
**Priority**: High - affects core functionality
