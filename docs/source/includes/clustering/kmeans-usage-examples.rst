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

Training
--------

::

   kmeans::model<> run_training(const table& data,
                              const table& initial_centroids) {
      const auto kmeans_desc = kmeans::descriptor<float>{}
         .set_cluster_count(10)
         .set_max_iteration_count(50)
         .set_accuracy_threshold(1e-4);

      const auto result = train(kmeans_desc, data, initial_centroids);

      print_table("labels", result.get_labels());
      print_table("centroids", result.get_model().get_centroids());
      print_value("objective", result.get_objective_function_value());

      return result.get_model();
   }

Inference
---------

::

   table run_inference(const kmeans::model<>& model,
                     const table& new_data) {
      const auto kmeans_desc = kmeans::descriptor<float>{}
         .set_cluster_count(model.get_cluster_count());

      const auto result = infer(kmeans_desc, model, new_data);

      print_table("labels", result.get_labels());
   }