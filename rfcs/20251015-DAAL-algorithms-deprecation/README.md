# Deprecation of certain DAAL algorithms in oneDAL 2025.10

## Introduction

The DAAL library was launched approximately 10 years ago with the goal
of providing comprehensive coverage for a wide variety of data analysis tasks.

Over the past decade, the data science field has evolved significantly,
and the relevance of certain algorithms has diminished.

Maintaining these less-utilized algorithms in oneDAL no longer provides
meaningful value to users while continuing to impose substantial
maintenance overhead.

As part of our long-term strategy to fully transition all functionality
to the modern oneDAL API, we are deprecating select DAAL algorithms
that no longer align with current needs.

This deprecation will deliver several benefits to oneDAL users:

 - Reduced binary size: Smaller, more efficient library distributions
 - Enhanced clarity: A more focused API that makes the library's capabilities
   easier to understand and navigate
 - Improved development efficiency: Faster build and testing cycles,
   particularly benefiting open source community contributors
 - Better resource allocation: Development resources can be redirected
   toward high-impact features and optimizations

## Proposal

We have identified two distinct tiers of algorithms scheduled for deprecation
based on their current relevance:

- **Tier 1**: Algorithms that demonstrate minimal user engagement,
  have declined in popularity within the data analytics ecosystem, and offer
  no significant performance advantages in oneDAL compared to alternative solutions.

- **Tier 2**: Algorithms that retain certain advantages in oneDAL,
  such as better performance or capabilities for processing large datasets
  through distributed computing, but lack oneDAL API integration and
  cannot be utilized with scikit-learn-intelex.

The following **Tier 1** algorithms will be marked as deprecated
in oneDAL 2025.10 with complete removal planned for the next major release,
oneDAL 2026.0:

### Analysis algorithms to be deprecated:
- [Association Rules](https://uxlfoundation.github.io/oneDAL/daal/algorithms/association_rules/association-rules.html)
- [Multivariate Outlier Detection](https://uxlfoundation.github.io/oneDAL/daal/algorithms/outlier_detection/multivariate.html)
- [Multivariate BACON Outlier Detection](https://uxlfoundation.github.io/oneDAL/daal/algorithms/outlier_detection/multivariate-bacon.html)
- [Univariate Outlier Detection](https://uxlfoundation.github.io/oneDAL/daal/algorithms/outlier_detection/univariate.html)
- [SAGA Optimization Solver](https://uxlfoundation.github.io/oneDAL/daal/algorithms/optimization-solvers/solvers/stochastic-average-gradient-accelerated-method.html)
- [SGD Optimization Solver](https://uxlfoundation.github.io/oneDAL/daal/algorithms/optimization-solvers/solvers/stochastic-gradient-descent-algorithm.html)
- [Adaptive Subgradient Method (AdaGrad)](https://uxlfoundation.github.io/oneDAL/daal/algorithms/optimization-solvers/solvers/adaptive-subgradient-method.html)
- [Sorting](https://uxlfoundation.github.io/oneDAL/daal/algorithms/sorting/index.html)
- [Quality Metrics](https://uxlfoundation.github.io/oneDAL/daal/algorithms/quality_metrics/index.html)
- [Quantile](https://uxlfoundation.github.io/oneDAL/daal/algorithms/quantiles/index.html)

### Training and prediction algorithms to be deprecated:
- [AdaBoost Classifier](https://uxlfoundation.github.io/oneDAL/daal/algorithms/boosting/adaboost.html)
- [AdaBoost Multiclass Classifier](https://uxlfoundation.github.io/oneDAL/daal/algorithms/boosting/adaboost-multiclass.html)
- [BrownBoost Classifier](https://uxlfoundation.github.io/oneDAL/daal/algorithms/boosting/brownboost.html)
- [LogitBoost Classifier](https://uxlfoundation.github.io/oneDAL/daal/algorithms/boosting/logitboost.html)
- [Classification Decision Tree](https://uxlfoundation.github.io/oneDAL/daal/algorithms/decision_tree/decision-tree-classification.html)
- [Regression Decision Tree](https://uxlfoundation.github.io/oneDAL/daal/algorithms/decision_tree/decision-tree-regression.html)
- [Naive Bayes Classifier](https://uxlfoundation.github.io/oneDAL/daal/algorithms/naive_bayes/naive-bayes-classifier.html)
- [Classificaton Stump](https://uxlfoundation.github.io/oneDAL/daal/algorithms/stump/classification.html)
- [Regression Stump](https://uxlfoundation.github.io/oneDAL/daal/algorithms/stump/regression.html)

**Tier 2** algorithms under consideration for future deprecation:

### Analysis algorithms to be deprecated:
- [Cholesky Decomposition](https://uxlfoundation.github.io/oneDAL/daal/algorithms/cholesky/cholesky.html)
- [QR Decomposition](https://uxlfoundation.github.io/oneDAL/daal/algorithms/qr/qr-decomposition.html)
- [Pivoted QR Decomposition](https://uxlfoundation.github.io/oneDAL/daal/algorithms/qr/qr-pivoted.html)
- [Singular Value Decomposition](https://uxlfoundation.github.io/oneDAL/daal/algorithms/svd/singular-value-decomposition.html)

### Training and prediction algorithms to be deprecated:
- [Implicit Alternating Least Squares](https://uxlfoundation.github.io/oneDAL/daal/algorithms/implicit_als/implicit-alternating-least-squares.html)

## Next Steps

**Tier 1** algorithms: Upon acceptance of this RFC, these algorithms
will be deprecated and subsequently removed in the upcoming major oneDAL
2026.0 release.

**Tier 2** algorithms: These will undergo individual community review
to determine deprecation status on a per-algorithm basis.
Implementation will proceed according to community decisions and feedback.
