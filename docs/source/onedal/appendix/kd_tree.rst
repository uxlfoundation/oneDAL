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

.. _kd_tree:

========
k-d Tree
========

:math:`k`-:math:`d` tree is a space-partitioning binary tree [Bentley80]_, where

- Each non-leaf node induces the hyperplane that splits the feature space into
  two parts. To define the splitting hyperplane explicitly, a non-leaf node
  stores the identifier of the feature (that defines axis in the feature space)
  and `a cut-point <kd_tree_cut_point_>`_

- Each leaf node of the tree has an associated subset (*a bucket*) of elements of
  the training data set. Feature vectors from a bucket belong to the region of
  the space defined by tree nodes on the path from the root node to the
  respective leaf.


-------------
Related terms
-------------

.. _kd_tree_cut_point:

A cut-point
   A feature value that corresponds to a non-leaf node of a :math:`k`-:math:`d`
   tree and defines the splitting hyperplane orthogonal to the axis specified by
   the given feature.
