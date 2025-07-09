# Discontinuing static-link library

## Introduction

When building oneDAL, it produces both shared libraries for dynamic linkage and
static-link objects (`.a`/`.lib`).

The produced builds are distributed through multiple channels, including PyPI,
conda-forge, Intel's conda channel, Spack, NuGet, offline installers for Windows
and Linux, and Intel repositories for Linux package managers like APT.

Some of those distributions include only the shared library - for example, there
are no static-link oneDAL libraries distributed in PyPI and NuGet due to file
size constraints, nor in Spack.

Most usage of oneDAL happens through the Python bindings with the extension for
Scikit-Learn, which only uses the shared library of oneDAL. Other known public
consumers of oneDAL, such as
[OAP MLlib](https://github.com/oap-project/oap-mllib) and
[ML.net](https://www.nuget.org/packages/Microsoft.ML.OneDal/) also use the
shared library, and do so in the way of creating another library with bindings
for oneDAL that are exposed to final users in more convenient languages.

We are not aware of any users of the static-link library of oneDAL, whether
public or private, and it is unclear whether there are good use-cases for a
static-link oneDAL library.

Procedures from oneDAL are unlikely to be used as building blocks in a broader
algorithm the way other libraries such as MKL would, and applications training
or serving machine learning models for tabular data typically do so from
interpreted languages like Python. Use-cases such as embedded devices, where
usage of C++ for serving machine learning models might be more common, are
unlikely to rely on algorithms for tabular data like the ones offered by oneDAL.

Yet, the static library is bundled in many distribution channels even if it
doesn't get used, increasing download sizes very significantly, and increasing
compilation times during development due to required production of these files.

## Proposal

The proposal here is to not produce static-link files for oneDAL at all, and to
discontinue the package `dal-static`.

Being a large breaking change, this would need to be done in a major release
such as 2026.0.

## Open Questions

* Do other UXL members rely on the static-link files?
* Are there any known consumers of the static-link oneDAL library?
* Would it be better to remove the static-link library altogether, or to leave
  it as a non-default option? It might be hard to test it properly if not built
  and distributed by default, but could be a reasonable compromise.
