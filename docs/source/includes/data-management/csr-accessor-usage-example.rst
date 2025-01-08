.. Copyright 2023 Intel Corporation
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

::

   #include <sycl/sycl.hpp>
   #include <iostream>

   #include "oneapi/dal/table/csr.hpp"
   #include "oneapi/dal/table/csr_accessor.hpp"

   namespace dal = oneapi::dal;

   int main() {
      sycl::queue queue { sycl::default_selector() };

      // Create arrays of data, column indices and row offsets of the table
      // in sparse CSR storage format with one-based indexing on host
      const float host_data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
      const std::int64_t host_column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
      const std::int64_t host_row_offsets[] = { 1, 4, 5, 7, 8 };

      constexpr std::int64_t row_count = 4;
      constexpr std::int64_t column_count = 4;
      constexpr std::int64_t element_count = 7;

      // Allocate SYCL shared memory for storing data, column indices and row offsets arrays
      auto shared_data = sycl::malloc_shared<float>(element_count, queue);
      auto shared_column_indices = sycl::malloc_shared<std::int64_t>(element_count, queue);
      auto shared_row_offsets = sycl::malloc_shared<std::int64_t>(row_count + 1, queue);

      // Copy data, column indices and row offsets arrays from host to SYCL shared memory
      auto data_event = queue.memcpy(shared_data, host_data, sizeof(float) * element_count);
      auto column_indices_event = queue.memcpy(shared_column_indices,
                                               host_column_indices,
                                               sizeof(std::int64_t) * element_count);
      auto row_offsets_event =
         queue.memcpy(shared_row_offsets, host_row_offsets, sizeof(std::int64_t) * (row_count + 1));

      auto table = dal::csr_table::wrap(queue,
                                        shared_data,
                                        shared_column_indices,
                                        shared_row_offsets,
                                        row_count,
                                        column_count,
                                        dal::sparse_indexing::one_based,
                                        { data_event, column_indices_event, row_offsets_event });

      // Accessing second and third rows of the table
      dal::csr_accessor<const float> acc{ table };

      const auto [block_data, block_column_indices, block_row_offsets] = acc.pull(queue, { 1, 3 });

      for (std::int64_t i = 0; i < block_data.get_count(); i++) {
         std::cout << block_data[i] << ", ";
      }
      std::cout << std::endl;

      for (std::int64_t i = 0; i < block_column_indices.get_count(); i++) {
         std::cout << block_column_indices[i] << ", ";
      }
      std::cout << std::endl;

      for (std::int64_t i = 0; i < block_row_offsets.get_count(); i++) {
         std::cout << block_row_offsets[i] << ", ";
      }
      std::cout << std::endl;

      sycl::free(shared_data, queue);
      sycl::free(shared_column_indices, queue);
      sycl::free(shared_row_offsets, queue);
      return 0;
   }