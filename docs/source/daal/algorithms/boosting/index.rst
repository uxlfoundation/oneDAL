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

Boosting
========

Boosting is a set of algorithms intended to build a strong classifier from an ensemble of weighted weak learners
by iterative re-weighting according to some accuracy measure for weak learners.
A weak learner is a classification or regression algorithm that has only slightly better performance than random guessing.
Weak learners are usually very simple and fast, and they focus on classification of very specific features.

Boosting algorithms include LogitBoost, BrownBoost, AdaBoost, and others.
A Decision Stump classifier is one of the popular weak learners.

In |short_name|, a weak learner is:

- Classification algorithm for AdaBoost and BrownBoost
- Regression algorithm for LogitBoost

Weak learners support training of the boosting model for weighted datasets.

|short_name| boosting algorithms pass pointers to weak learner training and prediction objects through the parameters of boosting algorithms.
Use the ``getNumberOfWeakLearners()`` method to determine the number of weak learners trained.

You can implement your own weak learners by deriving from the appropriate interface classes:

- Classification for AdaBoost and BrownBoost
- Regression for LogitBoost

.. note::

    When defining your own weak learners to use with boosting classifiers, make sure the prediction component of your weak learner returns:

    - The number from :math:`\{-1, 1\}` in case of binary classification.
    - Class label from :math:`\{0, \ldots, \text{nClasses}-1\}` for ``nClasses`` > 2.
    - Some boosting algorithms like SAMME.R AdaBoost that require probabilities of classes.
      For description of each boosting algorithm, refer to a corresponding section in this document.

.. toctree::
   :maxdepth: 1

   adaboost.rst
   adaboost-multiclass.rst
   brownboost.rst
   logitboost.rst
