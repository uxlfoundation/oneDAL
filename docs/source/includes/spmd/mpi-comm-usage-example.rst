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

::

   #ifndef ONEDAL_DATA_PARALLEL
   #define ONEDAL_DATA_PARALLEL
   #endif

   #include "oneapi/dal/algo/kmeans.hpp"
   #include "oneapi/dal/spmd/mpi/communicator.hpp"

   kmeans::model<> run_training(const table& data,
                              const table& initial_centroids) {
      const auto kmeans_desc = kmeans::descriptor<float>{}
         .set_cluster_count(10)
         .set_max_iteration_count(50)
         .set_accuracy_threshold(1e-4);

      auto comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::mpi>(queue);
      auto rank_id = comm.get_rank();

      const auto result_train = dal::preview::train(comm, kmeans_desc, local_input);

      if(rank_id == 0) {
         print_table("centroids", result.get_model().get_centroids());
         print_value("objective", result.get_objective_function_value());
      }
      return result.get_model();
   }