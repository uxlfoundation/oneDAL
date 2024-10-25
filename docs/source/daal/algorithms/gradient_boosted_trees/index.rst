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

Gradient Boosted Trees
======================

The library provides gradient boosted trees classification and
regression algorithms based on an ensemble of regression (decision)
trees trained using stochastic gradient boosting technique.
*Regression tree* is a binary tree graph. Its internal (split) nodes
represent a *decision function* used to select following (child) node
at prediction stage. Its leaf (terminal) nodes represent the
corresponding response values which are the result of prediction from
the tree. For more details, see Decision Tree [Breiman84]_.

.. toctree::
   :maxdepth: 1

   gradient-boosted-trees
   gradient-boosted-trees-regression
   gradient-boosted-trees-classification
