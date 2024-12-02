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

Computing
---------

::

   table run_compute(const table& data) {
      const auto kmeans_desc = kmeans_init::descriptor<float,
                                                      kmeans_init::method::dense>{}
         .set_cluster_count(10)

      const auto result = compute(kmeans_desc, data);

      print_table("centroids", result.get_centroids());

      return result.get_centroids();
   }