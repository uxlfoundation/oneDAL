# Static libraries handling Update(RFC)

### Revision
|Date       |Revision| Comments                                                                 |
|-----------|--------|--------------------------------------------------------------------------|
|  20250612 |  1.0   | Initial version                                                          |

## Motivation

* What problem do we solve?
  * **Significant reduction in library size**:  
    Removing symbol copying for static libraries will reduce the size of the oneDAL standalone package by approximately:
    - ~50% (~700 MB) on Linux
    - ~X% (~X GB) on Windows (TBD)
  * **Cleaner and more maintainable build system**:  
    Simplifies both Bazel and CMake logic by removing fragile `addlib`-style manual bundling.

* What is the customer impact if we don't do this?
  * Users may face issues with dependencies handling.

* What operations systems will be impacted?
  * We will have the same approach for both windows and linux systems.

* What is the timeline?
  * **Deprecation notice**: oneDAL 2025.7 release notes
  * **Symbol copying removal**: oneDAL 2026.0

## Initial Results of Proposed Changes

## Introduction

Currently, in the oneDAL build process, mathematical backends such as **OpenBLAS** for ARM and **Intel MKL** are linked by **copying all symbols** from the corresponding static libraries (e.g., `libopenblas.a`, `libmkl_core.a`) into the resulting archive via custom `addlib` logic in the `Makefile`.

This approach leads to bundling large static libraries into our own library archives, increasing binary size and leading to symbol duplication or conflicts if the final application also links to the same libraries.

## Proposal

We propose to change the structure of static library handling in oneDAL to address several growing concerns:

1. **Remove symbol duplication and overbundling**:  
   Eliminate the current `addlib` logic that embeds all symbols from backend libraries (e.g., MKL/OpenBLAS) directly into the oneDAL static archives.  
   Instead, treat these libraries as **link-time dependencies** only — through `target_link_libraries()` in CMake or `EXTRA_LIBS` in Makefile — shifting responsibility for final linkage to the user.  
   This avoids binary bloat, reduces symbol conflicts, and simplifies maintenance.

## Open Questions

- How will this change impact current downstream consumers who rely on bundled behavior?
- Should we maintain backward compatibility (e.g., via a CMake switch)?
- Should we provide guidance or tooling for users to configure correct BLAS/LAPACK backend linkage?

## oneDAL PR

https://github.com/uxlfoundation/oneDAL/pull/3237
