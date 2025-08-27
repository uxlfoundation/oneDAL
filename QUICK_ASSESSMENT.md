# Quick Assessment: oneDAL Copilot Integration Requirements

## 🎯 Main Goal: PR Reviews in GitHub

**Question**: Should anything be changed in instructions for PR review focus?

**Answer**: YES - Current instructions are generic. Need to add:
- PR review specific guidance
- Common review scenarios
- Code review best practices
- Validation checklists

## 🔧 Critical Issues Identified

### 1. Build System Priorities (INCORRECT)
**Current State**: Instructions say Bazel is primary
**Reality**: 
- **Make**: Primary platform for building product
- **CMake**: Only for end-user integration (find_package support)
- **Bazel**: Ongoing migration, used for new tests and development (NOT production builds)

**Impact**: This is a critical error that will mislead Copilot users

### 2. Missing scikit-learn-intelex Integration
**Current State**: No mention of working relationship
**Reality**: oneDAL works together with scikit-learn-intelex project
- Separate repositories but working together
- Common validation aspects
- Cross-repository considerations

**Impact**: Missing context for integration scenarios

### 3. Missing CI Infrastructure Documentation
**Current State**: No CI workflows documented
**Reality**: Need comprehensive CI documentation based on UXL Foundation patterns

## 📋 Immediate Action Items

### High Priority (Fix This Week)
1. **Correct Build System Priorities** in all instruction files
2. **Add PR Review Focus** sections
3. **Create Cross-References** between instructions and AGENTS.md files

### Medium Priority (Next Week)
1. **Add scikit-learn-intelex context**
2. **Create deploy/AGENTS.md**
3. **Streamline verbose examples**

### Low Priority (Following Week)
1. **Create CI workflows documentation**
2. **Create ci/AGENTS.md**

## 🔍 Specific Changes Needed

### Build System Corrections
```markdown
# WRONG (Current)
- **Bazel**: Primary build system for development and CI/CD
- **CMake**: Alternative build system for traditional workflows  
- **Make**: Legacy build system for specific platform requirements

# CORRECT (Required)
- **Make**: Primary build system for production builds
- **CMake**: End-user integration support (find_package)
- **Bazel**: Development and testing (ongoing migration)
```

### PR Review Focus Addition
```markdown
## PR Review Assistance

### Common Review Scenarios
- **Build System Changes**: Verify Make compatibility first
- **New Features**: Check interface consistency
- **Tests**: Validate Bazel test configuration
- **Documentation**: Ensure accuracy and completeness
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

## 📊 Impact Assessment

### Current Problems
- ❌ Copilot will suggest Bazel for production builds (wrong)
- ❌ Missing PR review context (main goal)
- ❌ No cross-repository integration context
- ❌ Examples too verbose for quick review

### After Fixes
- ✅ Copilot will suggest correct build system for context
- ✅ PR review assistance will be primary focus
- ✅ Integration scenarios will be properly covered
- ✅ Examples will be concise and review-friendly

## 🚀 Next Steps

1. **Approve this assessment**
2. **Begin Phase 1 implementation** (critical corrections)
3. **Test with actual PR review scenarios**
4. **Validate build system suggestions**
5. **Iterate based on feedback**

---

**Status**: Assessment complete, ready for implementation
**Priority**: HIGH - affects core functionality and main goal
**Timeline**: 1-2 weeks for critical fixes
