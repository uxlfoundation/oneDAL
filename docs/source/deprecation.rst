.. Copyright 2023 Intel Corporation
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.


Deprecation Notice
==================

This page provides information about the deprecations of a specific oneAPI Data Analytics Library (oneDAL) functionality.

Selected DAAL Algorithms
************************

**Deprecation:** Selected DAAL algorithms in the oneDAL library are marked as deprecated. The 2025.x releases are the last to provide it.
The list of the deprecated DAAL analysis algorithms includes:

- `Association Rules <daal_alg_association_rules>`_
- `Multivariate Outlier Detection <daal_alg_multivariate_outlier_detection>`_
- `Multivariate BACON Outlier Detection <daal_alg_multivariate_bacon_outlier_detection>`_
- `Univariate Outlier Detection <daal_alg_univariate_outlier_detection>`_
- `SAGA Optimization Solver <saga_solver>`_
- `SGD Optimization Solver <sgd_solver>`_
- `Adaptive Subgradient Method (AdaGrad) <adagrad_solver>`_
- `Sorting <daal_alg_sorting>`_
- `Quantile <daal_alg_quantile>`_

The list of the deprecated DAAL training and prediction algorithms includes:

- `AdaBoost Classifier <daal_alg_adaboost_classifier>`_
- `AdaBoost Multiclass Classifier <daal_alg_adaboost_multiclass_classifier>`_
- `BrownBoost Classifier <daal_alg_brownboost_classifier>`_
- `Decision Tree Classifier <_dt_classification>`_
- `Decision Tree Regressor <_dt_regression>`_
- `LogitBoost Classifier <daal_alg_logitboost_classifier>`_
- `Multinomial Naive Bayes Classifier <daal_alg_naive_bayes_classifier>`_
- `Stump Classifier <daal_alg_stump_classifier>`_
- `Stump Regressor <daal_alg_stump_regressor>`_

**Reasons for deprecation:** The ongoing efforts to optimize oneDAL resources and focus strongly on the most widely used features.

**Alternatives:** Consider using other libraries that provide similar functionality. For example, mlxtend for Association Rules: https://rasbt.github.io/mlxtend/

Selected DAAL Data Sources
**************************

**Deprecation:** Selected DAAL data sources in the oneDAL library are marked as deprecated. The 2025.x releases are the last to provide it.
The list of the deprecated DAAL data sources includes:

- KDB Data Source
- MySQL Data Source
- ODBC Data Source

**Reasons for deprecation:** The value of DAAL's native data source implementations has diminished as Python's ecosystem of data manipulation libraries,
including pandas, Parquet, and specialized database connectors, now provides improved flexibility and functionality for data import.

**Alternatives:** For data access, consider using other packages that provide similar functionality. For example, `SQLAlchemy <https://www.sqlalchemy.org/>`,
or `pyodbc <https://pypi.org/project/pyodbc/>` for Python users.

DAAL Matrices
*************

**Deprecation:** DAAL matrices in the oneDAL library are marked as deprecated. The 2025.x releases are the last to provide it.
The list of the deprecated DAAL classes includes:

- Matrix
- PackedSymmetricMatrix
- PackedTriangularMatrix

**Reasons for deprecation:** These data layouts have limited adoption outside of DAAL and are not widely used in the broader data science ecosystem.
Maintaining these specialized data layouts in oneDAL no longer provides meaningful value to users while continuing to impose substantial maintenance overhead.

**Alternatives:** Consider using HomogenNumericTable or oneapi::dal::homogen_table or other widely adopted data structures
such as NumPy arrays or pandas DataFrames for data representation in Python.

Java* Interfaces
****************

**Deprecation:** The Java* interfaces in the oneDAL library are marked as deprecated. The future releases of the oneDAL library may no longer include support for these Java* interfaces.

**Reasons for deprecation:** The ongoing efforts to optimize oneDAL resources and focus strongly on the most widely used features.

**Alternatives:** Intel(R) Optimized Analytics Package* (OAP) project for the Spark* users.
The project offers a comprehensive set of optimized libraries, including the OAP* MLlib* component. For more information, visit https://github.com/oap-project/oap-mllib.

Compression Functionality
*************************

**Deprecation:** The compression functionality in the oneDAL library is deprecated. Starting with the 2024.0 release, oneDAL will not support the compression functionality.

**Reasons for deprecation:** The ongoing efforts to optimize oneDAL resources and focus strongly on the most widely used features.

**Alternatives:** The external compression mechanics with optimized into your application implementation. For example, Intel(R) IPP.

ABI Compatibility
*****************

**Deprecation:** ABI compatibility is to be broken as part of the 2024.0 release of oneDAL. The library's major version is to be incremented to two to enforce the relinking of existing applications.

**Reasons for deprecation:**  The clean-up process of the deprecated functionality, interfaces, and symbols.

**Alternatives:** Relink to a newer version.

macOS* Support
**************

**Deprecation:** macOS* support is deprecated for oneDAL. The 2023.x releases are the last to provide it.

**Reasons for deprecation:**  No modern X86 macOS*-based systems are to be released.

**Alternatives:** The 2023.x version on macOS*.
