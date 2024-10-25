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

::

   #include <sycl/sycl.hpp>
   #include <iostream>

   #include "oneapi/dal/table/homogen.hpp"
   #include "oneapi/dal/table/column_accessor.hpp"

   using namespace oneapi;

   int main() {
      sycl::queue queue { sycl::default_selector() };

      constexpr float host_data[] = {
         1.0f, 1.5f, 2.0f,
         2.1f, 3.2f, 3.7f,
         4.0f, 4.9f, 5.0f,
         5.2f, 6.1f, 6.2f
      };

      constexpr std::int64_t row_count = 4;
      constexpr std::int64_t column_count = 3;

      auto shared_data = sycl::malloc_shared<float>(row_count * column_count, queue);
      auto event = queue.memcpy(shared_data, host_data, sizeof(float) * row_count * column_count);
      auto t = dal::homogen_table::wrap(queue, shared_data, row_count, column_count, { event });

      // Accessing whole elements in a first column
      dal::column_accessor<const float> acc { t };

      auto block = acc.pull(queue, 0);
      for(std::int64_t i = 0; i < block.get_count(); i++) {
         std::cout << block[i] << ", ";
      }
      std::cout << std::endl;

      sycl::free(shared_data, queue);
      return 0;
   }