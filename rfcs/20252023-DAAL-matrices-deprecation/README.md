# Deprecation of certain DAAL numeric tables in oneDAL 2025.1X

## Introduction

DAAL supports various in-memory data storage formats through its
[numeric table interface](https://uxlfoundation.github.io/oneDAL/daal/data-management/numeric-tables.html),
enabling efficient data access from different source types for processing
by computational algorithms.

However, some of these data layouts have limited adoption outside of DAAL
and are not widely used in the broader data science ecosystem.

Maintaining these specialized data layouts in oneDAL no longer provides
meaningful value to users while continuing to impose substantial maintenance
overhead.

## Proposal

The following types of [homogeneous numeric tables](https://uxlfoundation.github.io/oneDAL/daal/data-management/numeric-tables-types.html#homogeneous-numeric-tables)
will be marked as deprecated in oneDAL 2025.1X with complete removal planned
for the next major release, oneDAL 2026.0:

- [Matrix](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/data_management/data/matrix.h#L49)
- [PackedSymmetricMatrix](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/data_management/data/symmetric_matrix.h#L106)
- [PackedTriangularMatrix](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/data_management/data/symmetric_matrix.h#L801)
