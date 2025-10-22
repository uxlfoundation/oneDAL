# Deprecation of DAAL API layers of certain algorithms in oneDAL 2025.1X

## Introduction

Since the introduction of the modern oneDAL interface, certain functionality
now exists with dual APIs: the legacy DAAL interface and the newer oneDAL
interface.

Maintaining both APIs for the same functionality creates unnecessary complexity
and maintenance overhead.

It is now appropriate to remove DAAL API components that have equivalent
oneDAL implementations.

In contrast to [PR \#3400](https://github.com/uxlfoundation/oneDAL/pull/3400),
this RFC proposes removing only the API layers while preserving all underlying
algorithmic computational kernels.
These kernels remain valuable as they are currently used,
or may be used in the future, to perform CPU computations beneath the oneDAL
interface.

## Proposal

The following APIs will be marked as deprecated in the next minor oneDAL 2025.X,
with complete API removal scheduled for the next major release, oneDAL 2026.0:


#### [Distance Matrix](https://uxlfoundation.github.io/oneDAL/daal/algorithms/distance/index.html)

- [correlation_distance::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distance/correlation_distance.h#L55)
- [correlation_distance::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distance/correlation_distance.h#L90)
- [cosine_distance::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distance/cosine_distance.h#L55)
- [cosine_distance::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distance/cosine_distance.h#L90)

#### [DBSCAN](https://uxlfoundation.github.io/oneDAL/daal/algorithms/dbscan/index.html)

- [dbscan::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/dbscan/dbscan_batch.h#L56)
- [dbscan::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/dbscan/dbscan_batch.h#L87)

#### [Decision Forest](https://uxlfoundation.github.io/oneDAL/daal/algorithms/decision_forest/index.html)

- [decision_forest::classification::prediction::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/decision_forest/decision_forest_classification_predict.h#L75)
- [decision_forest::classification::prediction::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/decision_forest/decision_forest_classification_predict.h#L113)
- [decision_forest::classification::training::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/decision_forest/decision_forest_classification_training_batch.h#L56)
- [decision_forest::classification::training::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/decision_forest/decision_forest_classification_training_batch.h#L93)
- [decision_forest::regression::prediction::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/decision_forest/decision_forest_regression_predict.h#L53)
- [decision_forest::regression::prediction::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/decision_forest/decision_forest_regression_predict.h#L86)
- [decision_forest::regression::training::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/decision_forest/decision_forest_regression_training_batch.h#L59)
- [decision_forest::regression::training::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/decision_forest/decision_forest_regression_training_batch.h#L94)

#### [k Nearest Neighbors Classifier](https://uxlfoundation.github.io/oneDAL/daal/algorithms/k_nearest_neighbors/k-nearest-neighbors-knn-classifier.html)

- [bf_knn_classification::prediction::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/k_nearest_neighbors/bf_knn_classification_predict.h#L53)
- [bf_knn_classification::prediction::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/k_nearest_neighbors/bf_knn_classification_predict.h#L87)
- [bf_knn_classification::training::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/k_nearest_neighbors/bf_knn_classification_training_batch.h#L53)
- [bf_knn_classification::training::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/k_nearest_neighbors/bf_knn_classification_training_batch.h#L87)

- ...
