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

Compute
-------

::

   void run_compute(const table& data,
                              const table& weights) {
      double epsilon = 1.0;
      std::int64_t max_observations = 5;
      const auto dbscan_desc = kmeans::descriptor<float>{epsilon, max_observations}
         .set_result_options(dal::dbscan::result_options::responses);

      const auto result = compute(dbscan_desc, data, weights);

      print_table("responses", result.get_responses());
   }
