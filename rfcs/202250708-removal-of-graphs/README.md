# Removal of current Graph functionalities from oneDAL

## Introduction

OneDAL contains some data structures and algorithms for dealing with graphs:
https://github.com/uxlfoundation/oneDAL/tree/main/cpp/oneapi/dal/graph

These include, among others:
* A data structure for representation of graphs.
* A few standard graph routines such as finding shortest paths, connected
  components, and counting triangles.
* Jaccard similarity computations.
* A clustering algorithm based on Louvain modularity.

However, these functionalities are very underdeveloped - for example, they do
not contain the same kinds of optimizations used elsewhere in the library (like
compiler pragmas, intrinsics, blocking optimized for certain cache sizes, calls
to MKL functions that aren't in NetLib BLAS, etc.), and have no SYCL-based
implementations to run on GPUs or in SPMD mode.

These graph algorithms have been in preview mode for a while, and have not been
planned to be exposed from the Python side in daal4py, nor in sklearnex - should
be noted that scikit-learn itself does not offer this type of functionalities,
so there is no analog to patch from the sklearnex side either.

In their current state, it's unclear whether they are helpful at all. No
profiling experiments have been conducted so far, but given that they do not
employ the same kind of optimization tricks seen elsewhere in the library, it's
unlikely that we would find their performance to be competitive against other
optimized graph libraries for CPU. And in the case of GPU, there is the
SyGraph library on which we might focus our efforts on in the future instead:
https://github.com/unisa-hpc/SYgraph

There are no plans to develop these graph algorithms further to reach the point
where they could be taken out of preview, nor to produce SYCL versions of them
that would make them more useful. It doesn't make much sense to have things in
a state like this in oneDAL.

## Proposal

Proposal here is to remove the current implementations of graph data structures
and algorithms from the oneDAL library.

Since they are in the preview namespace, it should even be possible to remove
them in a minor version update, although it could wait for a major release if
needed.

## Open Questions

* Are graphs algorithms from oneDAL used in practice by any known consumer of
  the library?
* Are there any plans from other UXL members to spend resources on further
  development of graph algorithms on oneDAL?
* Are graph algorithms a topic of focus/interest to other UXL members?
