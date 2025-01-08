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

.. highlight:: cpp
.. default-domain:: cpp

.. _api_pca:

===================================
Principal Components Analysis (PCA)
===================================

.. include:: ../../../includes/decomposition/pca-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Principal Components Analysis <alg_pca>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::pca`` namespace and be available via inclusion of the
``oneapi/dal/algo/pca.hpp`` header file.

Enum classes
------------
.. onedal_enumclass:: oneapi::dal::pca::normalization

Descriptor
----------
.. onedal_class:: oneapi::dal::pca::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::pca::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::pca::task

Model
-----
.. onedal_class:: oneapi::dal::pca::model

.. _pca_t_api:

Training :cpp:expr:`train(...)`
--------------------------------
.. _pca_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::pca::train_input

.. _pca_t_api_result:

Result and Finalize Result
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::pca::train_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              pca::train_result train(const Descriptor& desc, \
                                         const pca::train_input& input)

   :param desc: PCA algorithm descriptor :expr:`pca::descriptor`
   :param input: Input data for the training operation

   Preconditions
      | :expr:`input.data.has_data == true`
      | :expr:`input.data.column_count >= desc.component_count`
   Postconditions
      | :expr:`result.means.row_count == 1`
      | :expr:`result.means.column_count == desc.component_count`
      | :expr:`result.variances.row_count == 1`
      | :expr:`result.variances.column_count == desc.component_count`
      | :expr:`result.variances[i] >= 0.0`
      | :expr:`result.eigenvalues.row_count == 1`
      | :expr:`result.eigenvalues.column_count == desc.component_count`
      | :expr:`result.model.eigenvectors.row_count == 1`
      | :expr:`result.model.eigenvectors.column_count == desc.component_count`

.. _pca_p_api:

Partial Training
----------------
.. _pca_p_api_input:

Partial Input
~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::pca::partial_train_input

.. _pca_p_api_result:

Partial Result and Finalize Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::pca::partial_train_result

.. _pca_f_api:

Finalize Training
-----------------

.. _pca_i_api:

Inference :cpp:expr:`infer(...)`
---------------------------------
.. _pca_i_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::pca::infer_input

.. _pca_i_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::pca::infer_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              pca::infer_result infer(const Descriptor& desc, \
                                         const pca::infer_input& input)

   :param desc: PCA algorithm descriptor :expr:`pca::descriptor`
   :param input: Input data for the inference operation

   Preconditions
      | :expr:`input.data.has_data == true`
      | :expr:`input.model.eigenvectors.row_count == desc.component_count`
      | :expr:`input.model.eigenvectors.column_count == input.data.column_count`
   Postconditions
      | :expr:`result.transformed_data.row_count == input.data.row_count`
      | :expr:`result.transformed_data.column_count == desc.component_count`

-------------
Usage Example
-------------

.. include:: ../../../includes/decomposition/pca-usage-examples.rst

--------
Examples
--------

.. include:: ../../../includes/decomposition/pca-examples.rst
