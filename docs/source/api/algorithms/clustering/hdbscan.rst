.. Copyright contributors to the oneDAL project
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

.. _api_hdbscan:

=======
HDBSCAN
=======

.. include:: ../../../includes/clustering/hdbscan-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: HDBSCAN <alg_hdbscan>`.

---------------------
Programming Interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::hdbscan`` namespace and are available via inclusion of the
``oneapi/dal/algo/hdbscan.hpp`` header file.

Descriptor
----------

::

   namespace oneapi::dal::hdbscan {

   template <typename Float = float,
             typename Method = method::brute_force,
             typename Task = task::clustering>
   class descriptor {
   public:
       descriptor(std::int64_t min_cluster_size = 5,
                  std::int64_t min_samples = 5);

       std::int64_t get_min_cluster_size() const;
       std::int64_t get_min_samples() const;
       distance_metric get_metric() const;
       double get_degree() const;
       result_option_id get_result_options() const;

       auto& set_min_cluster_size(std::int64_t value);
       auto& set_min_samples(std::int64_t value);
       auto& set_metric(distance_metric value);
       auto& set_degree(double value);
       auto& set_result_options(const result_option_id& value);
   };

   } // namespace oneapi::dal::hdbscan

Method tags
~~~~~~~~~~~

::

   namespace oneapi::dal::hdbscan::method {
       struct brute_force {};
       struct kd_tree {};
   }

Task tags
~~~~~~~~~

::

   namespace oneapi::dal::hdbscan::task {
       struct clustering {};
   }

Distance metrics
~~~~~~~~~~~~~~~~

::

   namespace oneapi::dal::hdbscan {
       enum class distance_metric {
           euclidean,
           manhattan,
           minkowski,
           chebyshev,
           cosine
       };
   }

.. _hdbscan_c_api:

Computation
-----------

.. _hdbscan_c_api_input:

Input
~~~~~

::

   namespace oneapi::dal::hdbscan {

   template <typename Task = task::clustering>
   class compute_input {
   public:
       compute_input(const table& data = {});

       const table& get_data() const;
       auto& set_data(const table& value);
   };

   } // namespace oneapi::dal::hdbscan

.. _hdbscan_c_api_result:

Result
~~~~~~

::

   namespace oneapi::dal::hdbscan {

   template <typename Task = task::clustering>
   class compute_result {
   public:
       const table& get_responses() const;
       std::int64_t get_cluster_count() const;
       const result_option_id& get_result_options() const;
   };

   } // namespace oneapi::dal::hdbscan

Operation
~~~~~~~~~

.. code-block:: cpp

   template <typename Descriptor>
   hdbscan::compute_result compute(const Descriptor& desc,
                                   const hdbscan::compute_input& input);

:param desc: HDBSCAN algorithm descriptor
:param input: Input data for the compute operation

Preconditions:
   - ``input.data.has_data == true``

--------
Examples
--------

.. include:: ../../../includes/clustering/hdbscan-examples.rst
