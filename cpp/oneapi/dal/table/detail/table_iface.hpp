/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "oneapi/dal/table/detail/access_iface_compat.hpp"
#include "oneapi/dal/table/detail/rows_access_iface.hpp"
#include "oneapi/dal/table/detail/columns_access_iface.hpp"
#include "oneapi/dal/table/detail/csr_access_iface.hpp"

#include "oneapi/dal/chunked_array.hpp"

namespace oneapi::dal {
namespace v1 {
class table_metadata;
enum class data_layout;
enum class feature_type;
enum class sparse_indexing;
} // namespace v1

using v1::table_metadata;
using v1::data_layout;
using v1::feature_type;
using v1::sparse_indexing;

} // namespace oneapi::dal

namespace oneapi::dal::detail {
namespace v1 {

// Inheritance from `access_provider_iface` is needed to support binary backward
// compatibility with the oneAPI Data Analytics Library (oneDAL) 2021.1. This should be removed in 2022.1.
class table_iface : public access_provider_iface {
public:
    virtual ~table_iface() = default;
    virtual std::int64_t get_column_count() const = 0;
    virtual std::int64_t get_row_count() const = 0;
    virtual std::int64_t get_kind() const = 0;
    virtual data_layout get_data_layout() const = 0;
    virtual const table_metadata& get_metadata() const = 0;
    virtual pull_rows_iface* get_pull_rows_iface() = 0;
    virtual pull_column_iface* get_pull_column_iface() = 0;
    virtual pull_csr_block_iface* get_pull_csr_block_iface() = 0;
};

class homogen_table_iface : public table_iface {
public:
    virtual dal::array<byte_t> get_data() const = 0;
};

class heterogen_table_iface : public table_iface {
public:
    //virtual heterogen_table get_row_slice(const range&) const = 0;
    virtual detail::chunked_array_base& get_column(std::int64_t) = 0;
    virtual const detail::chunked_array_base& get_column(std::int64_t) const = 0;
    virtual void set_column(std::int64_t, data_type, detail::chunked_array_base) = 0;
};

class csr_table_iface : public table_iface {
public:
    virtual dal::array<byte_t> get_data() const = 0;
    virtual dal::array<std::int64_t> get_column_indices() const = 0;
    virtual dal::array<std::int64_t> get_row_offsets() const = 0;
    virtual std::int64_t get_non_zero_count() const = 0;
    virtual sparse_indexing get_indexing() const = 0;
};

class table_builder_iface {
public:
    virtual ~table_builder_iface() = default;
    virtual table_iface* build() = 0;
    virtual pull_rows_iface* get_pull_rows_iface() = 0;
    virtual pull_column_iface* get_pull_column_iface() = 0;
    virtual push_rows_iface* get_push_rows_iface() = 0;
    virtual push_column_iface* get_push_column_iface() = 0;
    virtual pull_csr_block_iface* get_pull_csr_block_iface() = 0;
};

class homogen_table_builder_iface : public table_builder_iface {
public:
    virtual homogen_table_iface* build_homogen() = 0;

    virtual void set_data_type(data_type dt) = 0;
    virtual void set_layout(data_layout layout) = 0;
    virtual void set_feature_type(feature_type ft) = 0;

    virtual void reset(const dal::array<byte_t>& data,
                       std::int64_t row_count,
                       std::int64_t column_count) = 0;

    virtual void allocate(std::int64_t row_count, //
                          std::int64_t column_count) = 0;

    virtual void copy_data(const void* data, //
                           std::int64_t row_count,
                           std::int64_t column_count) = 0;

    virtual void copy_data(const dal::array<byte_t>& data) = 0;

#ifdef ONEDAL_DATA_PARALLEL
    virtual void allocate(const data_parallel_policy& policy,
                          std::int64_t row_count,
                          std::int64_t column_count,
                          sycl::usm::alloc alloc) = 0;

    virtual void copy_data(const data_parallel_policy& policy,
                           const void* data,
                           std::int64_t row_count,
                           std::int64_t column_count) = 0;
#endif
};

class heterogen_table_builder_iface : public table_builder_iface {
public:
    virtual heterogen_table_iface* build_heterogen() = 0;
};

class csr_table_builder_iface : public table_builder_iface {
public:
    virtual void set_data_type(data_type dt) = 0;

    virtual void reset(const dal::array<byte_t>& data,
                       const dal::array<std::int64_t>& column_indices,
                       const dal::array<std::int64_t>& row_offsets,
                       std::int64_t row_count,
                       std::int64_t column_count,
                       sparse_indexing indexing) = 0;
#ifdef ONEDAL_DATA_PARALLEL
    virtual void reset(const dal::array<byte_t>& data,
                       const dal::array<std::int64_t>& column_indices,
                       const dal::array<std::int64_t>& row_offsets,
                       std::int64_t row_count,
                       std::int64_t column_count,
                       sparse_indexing indexing,
                       const std::vector<sycl::event>& dependencies) = 0;
#endif

    virtual csr_table_iface* build_csr() = 0;
};

/// Generic table template is expected to implement all access interfaces to the table.
/// The example of the table that implements generic interface is the empty one.
template <typename Derived>
class generic_table_template : public table_iface,
                               public pull_rows_template<Derived>,
                               public pull_column_template<Derived>,
                               public pull_csr_block_template<Derived> {
public:
    pull_rows_iface* get_pull_rows_iface() override {
        return this;
    }

    pull_column_iface* get_pull_column_iface() override {
        return this;
    }

    pull_csr_block_iface* get_pull_csr_block_iface() override {
        return this;
    }
};

/// Homogen table template must implement row and column accessor, but not CSR.
template <typename Derived>
class homogen_table_template : public homogen_table_iface,
                               public pull_rows_template<Derived>,
                               public pull_column_template<Derived> {
public:
    pull_rows_iface* get_pull_rows_iface() override {
        return this;
    }

    pull_column_iface* get_pull_column_iface() override {
        return this;
    }

    pull_csr_block_iface* get_pull_csr_block_iface() override {
        return nullptr;
    }
};

/// Heterogen table template must implement row and column accessor, but not CSR.
template <typename Derived>
class heterogen_table_template : public heterogen_table_iface,
                                 public pull_rows_template<Derived>,
                                 public pull_column_template<Derived> {
public:
    pull_rows_iface* get_pull_rows_iface() override {
        return this;
    }

    pull_column_iface* get_pull_column_iface() override {
        return this;
    }

    pull_csr_block_iface* get_pull_csr_block_iface() override {
        return nullptr;
    }
};

/// CSR table template must implement CSR acessors, however row and column accessor are
/// optional and not assumed by default. At the same time the methods that returns
/// corresponding access interfaces may be overloaded in particualar CSR table implementation.
template <typename Derived>
class csr_table_template : public csr_table_iface, public pull_csr_block_template<Derived> {
public:
    pull_rows_iface* get_pull_rows_iface() override {
        return nullptr;
    }

    pull_column_iface* get_pull_column_iface() override {
        return nullptr;
    }

    pull_csr_block_iface* get_pull_csr_block_iface() override {
        return this;
    }
};

/// Homogen builder template must implement the same set of accessor as homogen table
/// template but also provide interfaces for write.
template <typename Derived>
class homogen_table_builder_template : public homogen_table_builder_iface,
                                       public pull_rows_template<Derived>,
                                       public pull_column_template<Derived>,
                                       public push_rows_template<Derived>,
                                       public push_column_template<Derived> {
public:
    pull_rows_iface* get_pull_rows_iface() override {
        return this;
    }

    pull_column_iface* get_pull_column_iface() override {
        return this;
    }

    push_rows_iface* get_push_rows_iface() override {
        return this;
    }

    push_column_iface* get_push_column_iface() override {
        return this;
    }

    pull_csr_block_iface* get_pull_csr_block_iface() override {
        return nullptr;
    }
};

template <typename Derived>
class heterogen_table_builder_template : public heterogen_table_builder_iface,
                                         public pull_rows_template<Derived>,
                                         public pull_column_template<Derived>,
                                         public push_column_template<Derived> {
public:
    pull_rows_iface* get_pull_rows_iface() override {
        return this;
    }

    pull_column_iface* get_pull_column_iface() override {
        return this;
    }

    push_column_iface* get_push_column_iface() override {
        return this;
    }

    pull_csr_block_iface* get_pull_csr_block_iface() override {
        return this;
    }
};

template <typename Derived>
class csr_table_builder_template : public csr_table_builder_iface,
                                   public pull_csr_block_template<Derived> {
public:
    pull_rows_iface* get_pull_rows_iface() override {
        return nullptr;
    }
    pull_column_iface* get_pull_column_iface() override {
        return nullptr;
    }
    push_rows_iface* get_push_rows_iface() override {
        return nullptr;
    }
    push_column_iface* get_push_column_iface() override {
        return nullptr;
    }
    pull_csr_block_iface* get_pull_csr_block_iface() override {
        return this;
    }
};

} // namespace v1

using v1::table_iface;
using v1::generic_table_template;
using v1::homogen_table_iface;
using v1::homogen_table_template;
using v1::heterogen_table_iface;
using v1::heterogen_table_template;
using v1::csr_table_iface;
using v1::csr_table_template;
using v1::table_builder_iface;
using v1::csr_table_builder_iface;
using v1::csr_table_builder_template;
using v1::homogen_table_builder_iface;
using v1::homogen_table_builder_template;
using v1::heterogen_table_builder_iface;
using v1::heterogen_table_builder_template;

} // namespace oneapi::dal::detail
