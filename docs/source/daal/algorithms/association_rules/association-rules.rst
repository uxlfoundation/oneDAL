.. Copyright 2019 Intel Corporation
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

Association Rules
=================

.. toctree::
   :glob:
   :maxdepth: 4

Association rules mining is the method for uncovering the most
important relationships between variables. Its main application is a
store basket analysis, which aims at discovery of a relationship
between groups of products with some level of confidence.


Details
*******

The library provides Apriori algorithm for association rule mining
[Agrawal94]_.

Let :math:`I = \{i_1, i_2, \ldots, i_m\}` be a set of items
(products) and subset :math:`T \subset I` is a transaction associated with item set
I. The association rule has the form: :math:`X \Rightarrow Y`, where :math:`X \subset I`, :math:`Y \subset I`, and
intersection of :math:`X` and :math:`Y` is empty: :math:`X \cap Y = \emptyset`. The left-hand-side set of
items (*itemset*) :math:`X` is called antecedent, while the right-hand-side
itemset Y is called consequent of the rule.

Let :math:`D = \{T_1, T_2, \ldots, T_n\}` be a set of
transactions, each associated with item set I. Item subset :math:`X \subset I` has
support :math:`s` in the transaction set :math:`D` if :math:`s` percent of transactions in :math:`D`
contains :math:`X`.

The association rule :math:`X \Rightarrow Y` in the transaction set :math:`D` holds with
confidence :math:`c` if :math:`c` percent of transactions in :math:`D` that contain :math:`X` also
contains :math:`Y`. Confidence of the rule can be represented as conditional
probability:

:math:`confidence(X \Rightarrow Y) = support (X \cup Y)/support(X)`

For a given set of transactions :math:`D = \{T_1, T_2, \ldots, T_n\}`, the minimum support s and minimum confidence c discover
all item sets :math:`X` with support greater than :math:`s` and generate all
association rules :math:`X \Rightarrow Y` with confidence greater than :math:`c`.

Therefore, the association rule discovery is decomposed into two
stages: mining (training) and discovery (prediction). The mining
stage involves generation of large item sets, that is, the sets that
have support greater than the given parameters. At the discovery
stage, the algorithm generates association rules using the large item
sets identified at the mining stage.

Batch Processing
****************

Algorithm Input
---------------

The association rules algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input
for your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Association Rules (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n \times 2` numeric table t with the mining data. Each row consists of two integers:

       + Transaction ID, the number between 0 and :math:`nTransactions - 1`.
       + Item ID, the number between 0 and :math:`nUniqueItems - 1`.

       The input can be an object of any class derived from NumericTable except PackedTriangularMatrix and PackedSymmetricMatrix.

Algorithm Parameters
--------------------

The association rules algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Association Rules (Batch Processing)
   :widths: 10 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - The computation method used by the algorithm. The only method supported so far is Apriori.
   * - ``minSupport``
     - :math:`0.01`
     - Minimal support, a number in the [0,1) interval.
   * - ``minConfidence``
     - :math:`0.6`
     - Minimal confidence, a number in the [0,1) interval.
   * - ``nUniqueItems``
     - :math:`0`
     - The total number of unique items. If set to zero, the library automatically determines the number of unique items from the input data.
   * - ``nTransactions``
     - :math:`0`
     - The total number of transactions. If set to zero, the library automatically determines the number transactions from the input data.
   * - ``discoverRules``
     - ``true``
     - A flag that enables generation of the rules from large item sets.
   * - ``itemsetsOrder``
     - ``itemsetsUnsorted``
     - The sort order of returned item sets:

       + itemsetsUnsorted - not sorted
       + itemsetsSortedBySupport - sorted by support in a descending order

   * - ``rulesOrder``
     - ``rulesUnsorted``
     - The sort order of returned rules:

       + rulesUnsorted - not sorted
       + rulesSortedByConfidence - sorted by support in a descending order

   * - ``minItemsetSize``
     - :math:`0`
     - A parameter that defines the minimal size of item sets to be included into the array of results. The value of zero imposes no limitations on the minimal size of item sets.
   * - ``maxItemsetSize``
     - :math:`0`
     - A parameter that defines the maximal size of item sets to be included into the array of results. The value of zero imposes no limitations on the maximal size of item sets.


Algorithm Output
----------------

The association rules algorithm calculates the result described
below. Pass the ``Result ID`` as a parameter to the methods that access
the results of your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Association Rules (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Result ID
     - Result
   * - ``largeItemsets``
     - Pointer to the numeric table with large item sets. The number of rows in
       the table equals the number of items in the large item sets. Each row
       contains two integers:

       + ID of the large item set, the number between 0 and nLargeItemsets -1.
       + ID of the item, the number between 0 and :math:`nUniqueItems-1`.

   * - ``largeItemsetsSupport``
     - Pointer to the :math:`nLargeItemsets \times 2` numeric table of support values. Each row contains two integers:

       + ID of the large item set, the number between 0 and nLargeItemsets-1.
       + The support value, the number of times the item set is met in the array of transactions.

   * - ``antecedentItemsets``
     - Pointer to the :math:`nAntecedentItems \times 2` numeric table that contains the
       left-hand-side (X) part of the association rules. Each row contains two integers:

       + Rule ID, the number between 0 and :math:`nAntecedentItems-1`.
       + Item ID, the number between 0 and :math:`nUniqueItems-1`.

   * - ``conseqentItemsets``
     - Pointer to the :math:`nConsequentItems \times 2` numeric table that contains the
       right-hand-side (Y) part of the association rules. Each row contains two integers:

       + Rule ID, the number between 0 and :math:`nConsequentItems-1`.
       + Item ID, the number between 0 and :math:`nUniqueItems-1`.

   * - ``confidence``
     - Pointer to the :math:`nRules \times 1` numeric table that contains confidence values
       of rules, floating-point numbers between 0 and 1. Confidence value in
       the i-th position corresponds to the rule with the index i.

By default, the result is an object of the HomogenNumericTable class,
but you can define the result as an object of any class derived from
NumericTable except PackedSymmetricMatrix, PackedTriangularMatrix, and
СSRNumericTable.

.. note::

   - The library requires transactions and items for each transaction
     to be passed in the ascending order.

   - Numbering of rules starts at 0.

   - The library calculates the sizes of numeric tables intended for
     results in a call to the algorithm. Avoid allocating the memory in
     numeric tables intended for results because, in general, it is
     impossible to accurately estimate the required memory size. If the
     memory interfaced by the numeric tables is allocated and its
     amount is insufficient to store the results, the algorithm returns
     an error.


Examples
--------

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`assoc_rules_apriori_batch.cpp <association_rules/assoc_rules_apriori_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`association_rules.py`

Performance Considerations
**************************

To get the best overall performance of the association rules
algorithm, whenever possible use the following numeric tables and
data types:

-  A SOA numeric table of type int to store features.

-  A homogenous numeric table of type int to store large item sets,
   support values, and left-hand-side and right-hand-side parts of
   association rules.

-  A numeric table with the confidence values of the same data type
   as specified in the algorithmFPType template parameter of the
   class.


.. include:: ../../../opt-notice.rst

