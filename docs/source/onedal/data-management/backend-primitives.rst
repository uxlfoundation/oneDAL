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

.. _dm_backend_primitives:

==================
Backend Primitives
==================

This section describes the types related to data management backend primitives.
Those primitives cannot be used directly in the user code but are used internally
by |short_name| to have a convenient and efficient way to work with data.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Data Management Backend Primitives
   :header-rows: 1
   :widths: 10 70
   :class: longtable

   * - Primitive
     - Description

   * - :ref:`api_ndorder`
     - An enumeration of multidimensional data orders used to store
       contiguous data blocks inside the table.

   * - :ref:`api_ndshape`
     - A class that represents the shape of a multidimensional array.

   * - :ref:`api_ndview`
     - An implementation of a multidimensional data container that provides a view of the homogeneous
       data stored in an externally-managed memory block.

   * - :ref:`api_ndarray`
     - A class that provides a way to store and manipulate homogeneous data
       in a multidimensional structure.

   * - :ref:`api_table2ndarray`
     - Functions that create ndarray objects from data tables. If possible, the data is stored in the
       same memory location as the data table and no data copying is performed.

-------------
Usage Example
-------------

The following listing provides an example of ndarray API to illustrate how data management backend
primitives can be used with |short_name| :txtref:`table` types to perform computations on the data:

.. code-block:: cpp

  namespace pr = dal::backend::primitives;

  // Create a 2D ndarray from input data table
  auto data_nd = pr::table2ndarray<float>(queue, data_table, sycl::usm::alloc::device);
  const std::int64_t row_count = data_table.get_row_count();
  const std::int64_t column_count = data_table.get_column_count();

  // Create a 1D ndarray to store the sum of each column
  auto sum_nd = pr::ndarray<float, 1>::empty(queue,
                                             { column_count },
                                             sycl::usm::alloc::device);

  // Get USM pointers to the data and sum arrays
  const float * data_ptr = data_nd.get_data();
  float * sum_ptr = sum_nd.get_mutable_data();

  constexpr std::int64_t row_block_size    = 1024;
  constexpr std::int64_t column_block_size = 512;

  // Define the SYCL* range for the kernel
  const auto range = sycl::nd_range<2>({ row_block_size, column_block_size }, { 1, column_block_size });

  // Compute the sum of each data row
  auto event = queue.submit([&](sycl::handler& cgh) {
        queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(range, [=](auto it) {
            const std::int64_t row_shift = it.get_global_id(0);
            const std::int64_t col_shift = it.get_local_id(1);
            for (auto row_idx = row_shift; row_idx < row_count; row_idx += row_block_size) {
                const auto start = row_idx * column_count;
                const auto end = start + column_count;
                // Exclusive storage of the partial sum for execution unit
                float local_sum = 0.0f;
                for (auto idx = start + col_shift; idx < end; idx += column_block_size) {
                    local_sum += data_ptr[idx];
                }
                // Reduction over the workgroup
                sum_ptr[row_idx] = sycl::reduce_over_group(it.get_group(), local_sum, sycl::plus<float>());
            }
        });
    });

---------------------
Programming interface
---------------------

Refer to :ref:`API: Data Management Backend Primitives <backend_primitives_programming_interface>`.
