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

.. _api_svm:

=======================================
Support Vector Machine Classifier (SVM)
=======================================

.. include:: ../../../includes/svm/svm-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Support Vector Machine Classifier <alg_svm>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::svm`` namespace and are available via inclusion of the
``oneapi/dal/algo/svm.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::svm::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::svm::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::svm::task

Model
-----
.. onedal_class:: oneapi::dal::svm::model

.. _svm_t_api:

Training :expr:`train(...)`
--------------------------------
.. _svm_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::svm::train_input


.. _svm_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::svm::train_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              svm::train_result train(const Descriptor& desc, \
                                      const svm::train_input& input)

   :param desc: SVM algorithm descriptor :expr:`svm::descriptor`.
   :param input: Input data for the training operation

   Preconditions
      | :expr:`input.data.is_empty == false`
      | :expr:`input.labels.is_empty == false`
      | :expr:`input.labels.column_count == 1`
      | :expr:`input.data.row_count == input.labels.row_count`

.. _svm_i_api:

Inference :expr:`infer(...)`
----------------------------
.. _svm_i_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::svm::infer_input


.. _svm_i_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::svm::infer_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              svm::infer_result infer(const Descriptor& desc, \
                                      const svm::infer_input& input)

   :param desc: SVM algorithm descriptor :expr:`svm::descriptor`.
   :param input: Input data for the inference operation

   Preconditions
      | :expr:`input.data.is_empty == false`

--------
Examples
--------

.. include:: ../../../includes/svm/svm-examples.rst
