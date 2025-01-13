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

   knn::model<> run_training(const table& data,
                           const table& labels) {
      const std::int64_t class_count = 10;
      const std::int64_t neighbor_count = 5;
      const auto knn_desc = knn::descriptor<float>{class_count, neighbor_count};

      const auto result = train(knn_desc, data, labels);

      return result.get_model();
   }

Inference
---------

::

   table run_inference(const knn::model<>& model,
                     const table& new_data) {
      const std::int64_t class_count = 10;
      const std::int64_t neighbor_count = 5;
      const auto knn_desc = knn::descriptor<float>{class_count, neighbor_count};

      const auto result = infer(knn_desc, model, new_data);

      print_table("labels", result.get_labels());
   }