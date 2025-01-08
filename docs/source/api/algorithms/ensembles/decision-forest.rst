.. Copyright 2021 Intel Corporation
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

.. default-domain:: cpp

.. _api_df:

==================================================
Decision Forest Classification and Regression (DF)
==================================================

.. include:: ../../../includes/ensembles/df-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Decision Forest Classification and Regression <alg_df>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::decision_forest`` namespace and are available via inclusion of the
``oneapi/dal/algo/decision_forest.hpp`` header file.

Enum classes
++++++++++++

`error_metric_mode`
~~~~~~~~~~~~~~~~~~~

error_metric_mode::none
   Do not compute error metric.

error_metric_mode::out_of_bag_error
   Train produces :math:`1 \times 1` table with cumulative prediction error for out of bag observations.

error_metric_mode::out_of_bag_error_per_observation
   Train produces :math:`n \times 1` table with prediction error for out-of-bag observations.

`variable_importance_mode`
~~~~~~~~~~~~~~~~~~~~~~~~~~

variable_importance_mode::none
   Do not compute variable importance.

variable_importance_mode::mdi
   Mean Decrease Impurity.
   Computed as the sum of weighted impurity decreases for all nodes where the variable is used, averaged over all trees in the forest.

variable_importance_mode::mda_raw
   Mean Decrease Accuracy (permutation importance).
   For each tree, the prediction error on the out-of-bag portion of the data is computed (error rate for classification, MSE for regression).
   The same is done after permuting each predictor variable. The difference between the two are then averaged over all trees.

variable_importance_mode::mda_scaled
   Mean Decrease Accuracy (permutation importance).
   This is MDA_Raw value scaled by its standard deviation.

`infer_mode`
~~~~~~~~~~~~

infer_mode::class_labels
   Infer produces a "math:`n \times 1` table with the predicted labels.

infer_mode::class_responses
   deprecated

infer_mode::class_probabilities
   Infer produces :math:`n \times c` table with the predicted class probabilities for each observation.

`voting_mode`
~~~~~~~~~~~~~

voting_mode::weighted
   The final prediction is combined through a weighted majority voting.

voting_mode::unweighted
   The final prediction is combined through a simple majority voting.

`splitter_mode`
~~~~~~~~~~~~~~~

splitter_mode::best
   The best splitting strategy chooses the best threshold for each feature while building trees
   in terms of impurity among all histogram bins and feature subsets.

splitter_mode::random
   The random splitting strategy chooses a random threshold for each feature while building trees
   and selects the best feature in terms of impurity
   computed for that random split from the feature subsets.

Descriptor
++++++++++
.. onedal_class:: oneapi::dal::decision_forest::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::decision_forest::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::decision_forest::task

Model
+++++
.. onedal_class:: oneapi::dal::decision_forest::model

.. _df_t_api:

Training :expr:`train(...)`
+++++++++++++++++++++++++++
.. _df_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::decision_forest::train_input


.. _df_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::decision_forest::train_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              decision_forest::train_result train(const Descriptor& desc, \
                                         const decision_forest::train_input& input)

   :param desc: Decision Forest algorithm descriptor :expr:`decision_forest::descriptor`.
   :param input: Input data for the training operation

   Preconditions
      | :expr:`input.data.is_empty == false`
      | :expr:`input.labels.is_empty == false`
      | :expr:`input.labels.column_count == 1`
      | :expr:`input.data.row_count == input.labels.row_count`
      | :expr:`desc.get_bootstrap() == true ||
               (desc.get_bootstrap() == false &&
                desc.get_variable_importance_mode() != variable_importance_mode::mda_raw &&
                desc.get_variable_importance_mode() != variable_importance_mode::mda_scaled)`
      | :expr:`desc.get_bootstrap() == true ||
               (desc.get_bootstrap() == false && desc.get_error_metric_mode() == error_metric_mode::none)`

.. _df_i_api:

Inference :expr:`infer(...)`
++++++++++++++++++++++++++++
.. _df_i_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::decision_forest::infer_input


.. _df_i_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::decision_forest::infer_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              decision_forest::infer_result infer(const Descriptor& desc, \
                                         const decision_forest::infer_input& input)

   :param desc: Decision Forest algorithm descriptor :expr:`decision_forest::descriptor`.
   :param input: Input data for the inference operation

   Preconditions
      | :expr:`input.data.is_empty == false`
