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

- Association Rules
- Multivariate Outlier Detection
- Multivariate BACON Outlier Detection
- Univariate Outlier Detection
- SAGA Optimization Solver
- SGD Optimization Solver
- Adaptive Subgradient Method (AdaGrad)
- Sorting
- Quality Metrics
- Quantile

The list of the deprecated DAAL training and prediction algorithms includes:

- AdaBoost Classifier
- AdaBoost Multiclass Classifier
- BrownBoost Classifier
- Decision Tree Classifier
- Decision Tree Regressor
- LogitBoost Classifier
- Multinomial Naive Bayes Classifier
- Stump Classifier
- Stump Regressor

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

**Alternatives:** For data access, consider using other packages that provide similar functionality. For example, `SQLAlchemy <https://www.sqlalchemy.org/>`_,
or `pyodbc <https://pypi.org/project/pyodbc/>`_ for Python users.

DAAL Matrices
*************

**Deprecation:** DAAL matrices in the oneDAL library are marked as deprecated. The 2025.x releases are the last to provide it.
The list of the deprecated DAAL classes includes:

- Matrix
- PackedSymmetricMatrix
- PackedTriangularMatrix

**Reasons for deprecation:** These data layouts have limited adoption outside of DAAL and are not widely used in the broader data science ecosystem.
Maintaining these specialized data layouts in oneDAL no longer provides meaningful value to users while continuing to impose substantial maintenance overhead.

**Alternatives:** Consider using ``HomogenNumericTable`` or ``oneapi::dal::homogen_table`` or other widely adopted data structures
such as NumPy arrays or pandas DataFrames for data representation in Python.

Interface layers of certain DAAL algorithms
*******************************************

**Deprecation:** ``Batch`` and ``BatchContainer`` classes of certain DAAL algorithms in the oneDAL library are marked as deprecated.
The 2025.x releases are the last to provide it.
The list of the deprecated DAAL classes includes:

:doc:`daal/algorithms/distributions/index`

- ``distributions::Input``
- ``distributions::Result``

- ``distributions::bernoulli::Batch``
- ``distributions::bernoulli::BatchContainer``
- ``distributions::bernoulli::Parameter``

- ``distributions::normal::Batch``
- ``distributions::normal::BatchContainer``
- ``distributions::normal::Parameter``

- ``distributions::uniform::Batch``
- ``distributions::uniform::BatchContainer``
- ``distributions::uniform::Parameter``

:doc:`daal/algorithms/engines/index`

- ``engines::Input``
- ``engines::Result``

- ``engines::mcg59::BatchContainer``
- ``engines::mcg59::Batch``

- ``engines::mrg32k3a::BatchContainer``
- ``engines::mrg32k3a::Batch``

- ``engines::mt19937::BatchContainer``
- ``engines::mt19937::Batch``

- ``engines::mt2203::BatchContainer``
- ``engines::mt2203::Batch``

- ``engines::philox4x32x10::BatchContainer``
- ``engines::philox4x32x10::Batch``

:doc:`daal/algorithms/kernel_function/kernel-functions`

- ``kernel_function::Input``
- ``kernel_function::Result``

- ``kernel_function::linear::BatchContainer``
- ``kernel_function::linear::Batch``
- ``kernel_function::linear::Input``

- ``kernel_function::rbf::BatchContainer``
- ``kernel_function::rbf::Batch``
- ``kernel_function::rbf::Input``

:doc:`contribution/cpu_features`

- ``CpuType``
- ``CpuFeature``

- ``AlgorithmDispatchContainer``
- ``AlgorithmDispatchContainer<batch, ...>``

- ``BatchContainer`` classes in all algorithms
- ``OnlineContainer`` classes in all algorithms
- ``DistributedContainer`` classes in all algorithms

**Reasons for deprecation:** These classes are intended for internal use and provide limited value to end users.
Maintaining these APIs in the public interface reduces product flexibility and clarity.
These API layers will be removed from DAAL API and moved to the ``daal::*::internal`` namespace while preserving
all underlying algorithmic computational kernels, as these kernels are used internally by other algorithms.

**Alternatives:** Use the oneDAL interface equivalents of the deprecated DAAL classes. For example,
use :ref:`engine_type <api_df>` parameter of the oneDAL Decision Forest algorithm instead of the deprecated
DAAL ``engines::mt19937::Batch`` class.

ABI Compatibility
*****************

**Deprecation:** ABI compatibility is to be broken as part of the 2024.0 and 2026.0 releases of oneDAL. The library's major version is to be incremented to two to enforce the relinking of existing applications.

**Reasons for deprecation:**  The clean-up process of the deprecated functionality, interfaces, and symbols.

**Alternatives:** Relink to a newer version.

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

macOS* Support
**************

**Deprecation:** macOS* support is deprecated for oneDAL. The 2023.x releases are the last to provide it.

**Reasons for deprecation:**  No modern X86 macOS*-based systems are to be released.

**Alternatives:** The 2023.x version on macOS*.
