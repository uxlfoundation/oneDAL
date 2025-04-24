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

Distributed Processing
**********************

.. note:: Distributed processing mode for Principal Component Analysis is not available on GPU.

This mode assumes that data set is split in nblocks blocks across computation nodes.

PCA computation in the distributed processing mode follows the general schema described in Algorithms.

Algorithm Parameters
--------------------

The PCA algorithm in the distributed processing mode has the following parameters, depending on the computation method parameter method:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.15}|\Y{0.55}|

.. list-table:: Algorithm Parameters for Principal Component Analysis (Distributed Processing)
   :widths: 10 10 10 30
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - Method
     - Default Value
     - Description
   * - ``computeStep``
     - ``defaultDense`` or ``svdDense``
     - Not applicable
     - The parameter required to initialize the algorithm. Can be:

       - ``step1Local`` - the first step, performed on local nodes
       - ``step2Master`` - the second step, performed on a master node
   * - ``algorithmFPType``
     - ``defaultDense`` or ``svdDense``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - Not applicable
     - ``defaultDense``
     - Available computation methods for PCA computation:

       - ``defaultDense`` - the correlation method
       - ``svdDense`` - the SVD method
   * - ``covariance``
     - ``defaultDense``
     - `SharedPtr<covariance::Distributed <computeStep, algorithmFPType, covariance::defaultDense> >`
     - The correlation and variance-covariance matrices algorithm to be used for PCA computations with the correlation method.
       For details, see :ref:`Correlation and Variance-covariance Matrices. Distributed Processing <cor_cov_distributed>`.

Use the following two-step schema:

.. _pca_step_1:

Step 1 - on Local Nodes
-----------------------

.. tabs::

    .. group-tab:: Correlation method (``defaultDense``)

        In this step, the PCA algorithm accepts the input described below.
        Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
        For more details, see :ref:`algorithms`.

        .. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

        .. list-table:: Input for Principal Component Analysis using Correlation method (Distributed Processing, Step 1)
            :widths: 10 60
            :header-rows: 1

            * - Input ID
              - Input
            * - ``data``
              - Pointer to the :math:`n_i \times p` numeric table that represents the :math:`i`-th data block on the local node.
                The input can be an object of any class derived from ``NumericTable``.

        In this step, PCA calculates the results described below.
        Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
        For more details, see :ref:`algorithms`.

        .. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

        .. list-table:: Output for Principal Component Analysis using Correlation method (Distributed Processing, Step 1)
            :widths: 10 60
            :header-rows: 1
            :class: longtable

            * - Result ID
              - Result
            * - ``nObservationsCorrelation``
              - Pointer to the :math:`1 \times 1` numeric table with the number of observations processed so far on the local node.

                .. note::
                    By default, this result is an object of the ``HomogenNumericTable`` class,
                    but you can define it as an object of any class derived from ``NumericTable`` except ``CSRNumericTable``.
            * - ``crossProductCorrelation``
              - Pointer to the :math:`p \times p` numeric table with the cross-product matrix computed so far on the local node.

                .. note::
                    By default, this table is an object of the ``HomogenNumericTable`` class,
                    but you can define it as an object of any class derived from ``NumericTable``
                    except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.
            * - ``sumCorrelation``
              - Pointer to the :math:`1 \times p` numeric table with partial sums computed so far on the local node.

                .. note::
                    By default, this table is an object of the ``HomogenNumericTable`` class,
                    but you can define it as an object of any class derived from ``NumericTable``
                    except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

    .. group-tab:: SVD method (``svdDense``)

        In this step, the PCA algorithm accepts the input described below.
        Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
        For more details, see :ref:`algorithms`.

        .. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

        .. list-table:: Input for Principal Component Analysis using SVD method (Distributed Processing, Step 1)
            :widths: 10 60
            :header-rows: 1

            * - Input ID
              - Input
            * - ``data``
              - Pointer to the :math:`n_i \times p` numeric table that represents the :math:`i`-th data block on the local node.
                The input can be an object of any class derived from ``NumericTable``.

        In this step, PCA calculates the results described below.
        Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
        For more details, see :ref:`algorithms`.

        .. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

        .. list-table:: Output for Principal Component Analysis using SVD method (Distributed Processing, Step 1)
            :widths: 10 60
            :header-rows: 1
            :class: longtable

            * - Result ID
              - Result
            * - ``nObservationsCorrelation``
              - Pointer to the :math:`1 \times 1` numeric table with the number of observations processed so far on the local node.

                .. note::
                    By default, this result is an object of the ``HomogenNumericTable`` class,
                    but you can define it as an object of any class derived from ``NumericTable`` except ``CSRNumericTable``.
            * - ``sumSVD``
              - Pointer to the :math:`1 \times p` numeric table with partial sums computed so far on the local node.

                .. note::
                    By default, this table is an object of the ``HomogenNumericTable`` class,
                    but you can define it as an object of any class derived from ``NumericTable``
                    except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.
            * - ``sumSquaresSVD``
              - Pointer to the :math:`1 \times p` numeric table with partial sums of squares computed so far on the local node.

                .. note::
                    By default, this table is an object of the ``HomogenNumericTable`` class,
                    but you can define it as an object of any class derived from ``NumericTable``
                    except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.
            * - ``auxiliaryDataSVD``
              - A collection of numeric tables each with the partial result to transmit to the master node for :ref:`Step 2 <pca_step_2>`.

                .. note::
                    The collection can contain objects of any class derived from ``NumericTable``
                    except the ``PackedSymmetricMatrix`` and ``PackedTriangularMatrix``.

.. _pca_step_2:

Step 2 - on Master Node
-----------------------

.. tabs::

    .. group-tab:: Correlation method (``defaultDense``)

        In this step, the PCA algorithm accepts the input described below.
        Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
        For more details, see :ref:`algorithms`.

        .. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

        .. list-table:: Input for Principal Component Analysis using Correlation method (Distributed Processing, Step 2)
            :widths: 10 60
            :header-rows: 1

            * - Input ID
              - Input
            * - ``partialResults``
              - A collection that contains results computed in :ref:`Step 1 <pca_step_1>` on local nodes
                (``nObservationsCorrelation``, ``crossProductCorrelation``, and ``sumCorrelation``).

                .. note::
                    The collection can contain objects of any class derived from ``NumericTable``
                    except the ``PackedSymmetricMatrix`` and ``PackedTriangularMatrix``.

        In this step, PCA calculates the results described below.
        Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
        For more details, see :ref:`algorithms`.

        .. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

        .. list-table:: Output for Principal Component Analysis using Correlation method (Distributed Processing, Step 2)
            :widths: 10 60
            :header-rows: 1
            :class: longtable

            * - Result ID
              - Result
            * - ``eigenvalues``
              - Pointer to the :math:`1 \times p` numeric table that contains eigenvalues in the descending order.
            * - ``eigenvectors``
              - Pointer to the :math:`p \times p` numeric table that contains eigenvectors in the row-major order.

        .. note::
            By default, these results are object of the ``HomogenNumericTable`` class,
            but you can define the result as an object of any class derived from ``NumericTable``
            except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

    .. group-tab:: SVD method (``svdDense``)

        In this step, the PCA algorithm accepts the input described below.
        Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
        For more details, see :ref:`algorithms`.

        .. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

        .. list-table:: Input for Principal Component Analysis using SVD method (Distributed Processing, Step 2)
            :widths: 10 60
            :header-rows: 1

            * - Input ID
              - Input
            * - ``partialResults``
              - A collection that contains results computed in :ref:`Step 1 <pca_step_1>` on local nodes
                (``nObservationsSVD``, ``sumSVD``, ``sumSquaresSVD``, and ``auxiliaryDataSVD``).

                .. note::
                    The collection can contain objects of any class derived from ``NumericTable``
                    except the ``PackedSymmetricMatrix`` and ``PackedTriangularMatrix``.

        In this step, PCA calculates the results described below.
        Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
        For more details, see :ref:`algorithms`.

        .. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

        .. list-table:: Output for Principal Component Analysis using SVD method (Distributed Processing, Step 2)
            :widths: 10 60
            :header-rows: 1
            :class: longtable

            * - Result ID
              - Result
            * - ``eigenvalues``
              - Pointer to the :math:`1 \times p` numeric table that contains eigenvalues in the descending order.
            * - ``eigenvectors``
              - Pointer to the :math:`p \times p` numeric table that contains eigenvectors in the row-major order.

        .. note::
            By default, these results are object of the ``HomogenNumericTable`` class,
            but you can define the result as an object of any class derived from ``NumericTable``
            except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.
