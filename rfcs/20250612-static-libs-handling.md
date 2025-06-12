# Static Libraries Handling (RFC)

## Introduction

Currently, in the oneDAL build process, mathematical backends such as **OpenBLAS** for ARM and **Intel MKL** are linked by **copying all symbols** from the corresponding static libraries (e.g., `libopenblas.a`, `libmkl_core.a`) into the resulting archive via custom `addlib` logic in `Makefile`.  

This approach leads to bundling large static libraries into our own library archives, increasing binary size and leading to symbol duplication or conflicts if the final application also links to the same libraries.

**Motivation** for this proposal:

- Improve maintainability and transparency in linking behavior.
- Reduce binary bloat in the resulting libraries.
- Improve compatibility with user applications that may want to choose or override the BLAS/LAPACK backend.
- Better separation of concerns: the final application should own the linkage of external dependencies.


## Proposal

The proposal is to change the current behavior of bundling static backend libraries (OpenBLAS, MKL) directly into our own archive/library, and instead:

- Declare those libraries as **link-time dependencies**, e.g., via `target_link_libraries()` in CMake or `EXTRA_LIBS` in Makefile.
- Ensure that consumers of the library (e.g., applications linking to our `libonedal_core.a`) are responsible for linking against the correct backend.


## Open Questions

List here the significant uncertainties that are relevant to the proposal.

However, sometimes the answers on these questions won't be found even if the
proposal is accepted, say, by leaving the current implementation as is.