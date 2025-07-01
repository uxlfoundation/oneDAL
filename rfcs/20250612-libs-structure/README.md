# Libraries Structure and Dependency Update(RFC)

### Revision
|Date       |Revision| Comments                                                                 |
|-----------|--------|--------------------------------------------------------------------------|
|  20250612 |  1.0   | Initial version                                                          |
|  20250701 |  1.1   | Static sycl drop updates                                                 |

## Motivation

* What problem do we solve?
  * **Significant reduction in library size**:  
    Removing symbol copying for static libraries will reduce the size of the oneDAL standalone package by approximately:
    - ~50% (~700 MB) on Linux
    - ~X% (~X GB) on Windows (TBD)
  * **Improved build and test efficiency**:  
    Avoids long link times associated with large static archives, especially for internal testing and CI systems.
  * **Alignment with Intel oneMKL roadmap**:  
    Static SYCL libraries will be deprecated in upcoming oneMKL releases.  
    Continuing support for them would require maintaining legacy configurations and diverging from upstream support.
  * **Cleaner and more maintainable build system**:  
    Simplifies both Bazel and CMake logic by removing fragile `addlib`-style manual bundling.

* What is the customer impact if we don't do this?
  * Users may face long build times, symbol conflicts, and compatibility issues with newer oneMKL versions.
  * Continued dependency on deprecated SYCL static libraries will make integration harder with future toolchains.

* What is the timeline?
  * **Deprecation notice**: oneDAL 2025.7 release notes
  * **Symbol copying removal**: oneDAL 2026.0
  * **Static SYCL drop**: oneDAL 2026.0

## Initial Results of Proposed Changes

|Branch     | Size   | Building time                | DPC examples/test bazel linking           |
|-----------|--------|------------------------------|-------------------------------------------|
|   main    |  1.5gb | 4min 38s -j 112              | 55 s                                      |
|   custom  |  710mb*| 3min 13s -j 112              | 15 s                                      |
*may be changed in the final version

## Introduction

Currently, in the oneDAL build process, mathematical backends such as **OpenBLAS** for ARM and **Intel MKL** are linked by **copying all symbols** from the corresponding static libraries (e.g., `libopenblas.a`, `libmkl_core.a`) into the resulting archive via custom `addlib` logic in the `Makefile`.

This approach leads to bundling large static libraries into our own library archives, increasing binary size and leading to symbol duplication or conflicts if the final application also links to the same libraries.

In addition, the upcoming oneMKL releases will **remove support for static SYCL libraries** (`libmkl_sycl*.a`). To remain compatible with the future oneMKL toolchain and simplify our own build system, we propose to **drop static SYCL support in oneDAL** as well.

This means:
- **SYCL-based components** of oneDAL (e.g., `libonedal_dpc`) will be **provided only as shared libraries (.so/.dll)** starting from version 2026.0.
- **Non-SYCL components** of oneDAL will continue to support **both static and dynamic linkage**, as they do today.

The goal is to:
- Reduce maintenance burden
- Align with Intel’s toolchain direction
- Prevent user-side issues with incompatible linkage setups

## Proposal

We propose to change the structure of static library handling in oneDAL to address several growing concerns:

1. **Remove symbol duplication and overbundling**:  
   Eliminate the current `addlib` logic that embeds all symbols from backend libraries (e.g., MKL/OpenBLAS) directly into the oneDAL static archives.  
   Instead, treat these libraries as **link-time dependencies** only — through `target_link_libraries()` in CMake or `EXTRA_LIBS` in Makefile — shifting responsibility for final linkage to the user.  
   This avoids binary bloat, reduces symbol conflicts, and simplifies maintenance.

2. **Drop static SYCL backend support completely**:  
   Starting from oneDAL 2026.0, we propose to **fully remove static SYCL linkage support**.  
   The motivation is twofold:
   - **oneMKL has announced deprecation** of static SYCL libraries and will not provide them in future releases.  
     Relying on these unsupported configurations poses a long-term maintenance risk and breaks forward compatibility.
   - Dynamic SYCL linkage provides better developer experience:
     - Faster incremental builds
     - Smaller binaries
     - Better compatibility with Intel DPC++ compiler and runtime deployment

   This means that going forward, **`libonedal_dpc.a` will no longer be available** — instead, only dynamic SYCL (`libonedal_sycl.so`) will be supported.  
   Any applications depending on static SYCL will need to migrate to dynamic linkage.


## Open Questions

- How will this change impact current downstream consumers who rely on bundled behavior?
- Should we maintain backward compatibility (e.g., via a CMake switch)?
- Should we provide guidance or tooling for users to configure correct BLAS/LAPACK backend linkage?

## oneDAL PR

https://github.com/uxlfoundation/oneDAL/pull/3237
