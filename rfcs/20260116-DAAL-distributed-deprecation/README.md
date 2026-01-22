# Deprecation of DAAL distributed functionality in oneDAL 2025.11

## Introduction

The distributed algorithms in DAAL are planned for migration to the oneDAL interface.
As part of the ongoing transition from the legacy DAAL to the modern oneDAL API,
the DAAL APIs for distributed algorithms will be deprecated and subsequently removed in the 2027.0 release.

## Proposal

The following APIs will be marked as deprecated in the next minor oneDAL 2025.11
with complete API removal scheduled for the major release, oneDAL 2027.0:

- [Correlation and Variance-Covariance Matrices](https://uxlfoundation.github.io/oneDAL/daal/algorithms/covariance/computation-distributed.html)
- [K-Means Clustering](https://uxlfoundation.github.io/oneDAL/daal/algorithms/kmeans/computation-distributed.html)
- [Linear and Ridge Regression](https://uxlfoundation.github.io/oneDAL/daal/algorithms/linear_ridge_regression/linear-ridge-regression-computation.html#distributed-processing)
- [Moments of Low Order](https://uxlfoundation.github.io/oneDAL/daal/algorithms/moments/computation-distributed.html)
- [Principal Component Analysis](https://uxlfoundation.github.io/oneDAL/daal/algorithms/pca/computation-distributed.html)
- [QR Decomposition](https://uxlfoundation.github.io/oneDAL/daal/algorithms/qr/without-pivoting/computation-distributed.html)
- [Singular Value Decomposition](https://uxlfoundation.github.io/oneDAL/daal/algorithms/svd/computation-distributed.html)

## Next Steps

Upon acceptance of this RFC, these algorithms will be deprecated and subsequently moved
to the ``internal`` namespace in the 2026.0 release, providing users with time to prepare
for the transition.
The complete removal will occur in one of 2026.X releases or in the 2027.0 release, but not before
the corresponding oneDAL alternatives have been fully implemented.
