.. Copyright 2020 Intel Corporation
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

Principal Components Analysis Transform
=======================================

The PCA transform algorithm transforms the data set to principal components.

Details
*******

Given a transformation matrix :math:`T` computed by :ref:`PCA <pca>` (eigenvectors in row-major order)
and data set :math:`X` as input, the PCA Transform algorithm transforms input data set :math:`X`
of size :math:`n \times p` to the data set :math:`Y` of size :math:`n \times p_r`, :math:`pr \leq p`.

Batch Processing
****************

Algorithm Input
---------------

The PCA Transform algorithm accepts the input described below.
Pass the ```Input ID``` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Principal Components Analysis Transform (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Input ID
     - Input
   * - ``data``
     - Use when the input data is a normalized or non-normalized data set.

       Pointer to the :math:`n \times p` numeric table that contains the input data set.
       This input can be an object of any class derived from ``NumericTable``.
   * - ``eigenvectors``
     - Principal components computed using the PCA algorithm.

       Pointer to the :math:`p_r \times p` numeric table :math:`(p_r \leq p)`.
       You can define it as an object of any class derived from ``NumericTable``,
       except for ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.
   * - ``dataForTransform``
     - Optional. Pointer to the key value-data collection containing the following data for PCA. The collection contains the following key-value pairs:

       mean
           means
       variance
           variances
       eigenvalue
           eigenvalues

       .. note::

        - If you do not provide the collection, the library will not apply the corresponding centering, normalization or whitening operation.
        - If one of the numeric tables in collection is ``NULL``, the corresponding operation will not be applied: centering for means, normalization for variances, whitening for eigenvalues.
        - If mean or variance tables exist, it should be a pointer to the :math:`1 \times p` numeric table.
        - If eigenvalue table is not ``NULL``, it is the pointer to (:math:`1 \times \text{nColumns}`) numeric table, where the number of the columns is greater than or equal to ``nComponents``.

Algorithm Parameters
--------------------

The PCA Transform algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.15}|\Y{0.55}|

.. list-table:: Algorithm Parameters for Principal Components Analysis Transform (Batch Processing)
   :header-rows: 1
   :widths: 10 10 10 60
   :align: left
   :class: longtable

   * - Parameter
     - method
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``defaultDense`` or ``svdDense``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``nComponents``
     - ``defaultDense``
     - :math:`0`
     - The number of principal components :math:`(p_r \leq p)`. If zero, the algorithm will compute the result for :math:`\text{nComponents} = p_r`.


Algorithm Output
----------------

The PCA Transform algorithm calculates the results described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Principal Components Analysis Transform (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Result ID
     - Result
   * - ``transformedData``
     - Pointer to the :math:`n \times p_r` numeric table that contains data projected to the principal components basis.

       .. note::

            By default, this result is an object of the ``HomogenNumericTable`` class, but you can define the result as an object
            of any class derived from ``NumericTable`` except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

Examples
*********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`pca_transform_dense_batch.cpp <pca_transform/pca_transform_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`pca_transform.py`
