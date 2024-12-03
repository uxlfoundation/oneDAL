.. ******************************************************************************
.. * Copyright 2020 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

Engines
=======

Random number engines are used for uniformly distributed random numbers generation by using a seed - the initial
value that allows to select a particular random number sequence.
Initialization is an engine-specific procedure.

.. rubric:: Algorithm Input

Engines accept the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Engines
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``tableToFill``
     - Pointer to the numeric table of size :math:`n \times p`.

       This input can be an object of any class derived from ``NumericTable``
       except ``CSRNumericTable``, ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``,
       and ``MergedNumericTable`` when it holds one of the above table types.

.. rubric:: Algorithm Output

Engines calculate the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Engines
   :widths: 10 60
   :header-rows: 1

   * - Result ID
     - Result
   * - ``randomNumbers``
     - Pointer to the :math:`n \times p` numeric table with generated random floating-point values of single or double precision.

       In |short_name|, engines are in-place, which means that the algorithm does not allocate memory for the distribution result,
       but returns pointer to the filled input.

.. note:: In the current version of the library, engines are used for random number generation only as a parameter of another algorithm.

.. rubric:: Parallel Random Number Generation

The following methods that support generation of sequences of random numbers in parallel are supported in library:

Family
    Engines follow the same algorithmic scheme with different algorithmic parameters.
    The set of the parameters guarantee independence of random number sequences produced by the engines.

    The example below demonstrates the idea for the case when 2 engines from the same family are used to generate 2 random sequences:

    .. figure:: images/englines-family-method-example.jpg
        :width: 300
        :alt: Generating two sequences independently with two engines

        Family method of random sequence generation

SkipAhead
    This method skips ``nskip`` elements of the original random sequence.
    This method allows to produce ``nThreads`` non-overlapping subsequences.

    The example below demonstrates the idea for the case when 2 subsequences are used from the random sequence:

    .. figure:: images/englines-skipahead-method-example.jpg
        :width: 300
        :alt: Generating a subsequence by skipping nSkip elements

        SkipAhead method of random sequence generation

LeapFrog
    This method generates random numbers with a stride of ``nThreads``.
    ``threadIdx`` is an index of the current thread.

    The example below demonstrates the idea for the case when 2 subsequences are used from the random sequence:

    .. figure:: images/englines-leapfrog-method-example.jpg
        :width: 300
        :alt: Generating a subsequence with stride=2

        LeapFrog method of random sequence generation

These methods are represented with member functions of classes that represent functionality described in the Engines section. See API References for details.

.. note:: Support of these methods is engine-specific.

.. toctree::
    :maxdepth: 1

    mt19937.rst
    mcg59.rst
    mrg32k3a.rst
    philox4x32x10.rst
    mt2203.rst
