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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/detail/table_kinds.hpp"
#include "oneapi/dal/table/backend/empty_table_impl.hpp"
#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal {
namespace detail {
namespace v1 {

class table_metadata_impl {
public:
    virtual ~table_metadata_impl() = default;
    virtual std::int64_t get_feature_count() const = 0;
    virtual const feature_type& get_feature_type(std::int64_t index) const = 0;
    virtual const data_type& get_data_type(std::int64_t index) const = 0;
    virtual const dal::array<feature_type>& get_feature_types() const = 0;
    virtual const dal::array<data_type>& get_data_types() const = 0;
    virtual alloc_kind get_alloc_kind() const = 0;
};

} // namespace v1
} // namespace detail

namespace v1 {

using detail::v1::table_metadata_impl;

class empty_metadata_impl : public table_metadata_impl,
                            public ONEDAL_SERIALIZABLE(empty_table_metadata_id) {
public:
    std::int64_t get_feature_count() const override {
        return detail::get_empty_table_kind();
    }

    const dal::array<data_type>& get_data_types() const override {
        static const dal::array<data_type> dtypes{};
        return dtypes;
    }

    const dal::array<feature_type>& get_feature_types() const override {
        static const dal::array<feature_type> ftypes{};
        return ftypes;
    }

    const feature_type& get_feature_type(std::int64_t) const override {
        throw domain_error(
            dal::detail::error_messages::cannot_get_feature_type_from_empty_metadata());
    }

    const data_type& get_data_type(std::int64_t) const override {
        throw domain_error(dal::detail::error_messages::cannot_get_data_type_from_empty_metadata());
    }

    alloc_kind get_alloc_kind() const override {
        return alloc_kind::non_usm;
    }

    void serialize(detail::output_archive& ar) const override {
        // Nothing to serialize
    }

    void deserialize(detail::input_archive& ar) override {
        // Nothing to deserialize
    }
};
ONEDAL_REGISTER_SERIALIZABLE(empty_metadata_impl)

class simple_metadata_impl : public table_metadata_impl,
                             public ONEDAL_SERIALIZABLE(simple_table_metadata_id) {
public:
    simple_metadata_impl() = default;

    simple_metadata_impl(const dal::array<data_type>& dtypes,
                         const dal::array<feature_type>& ftypes,
                         const alloc_kind kind)
            : dtypes_(dtypes),
              ftypes_(ftypes),
              kind_(kind) {
        if (dtypes_.get_count() != ftypes_.get_count()) {
            throw out_of_range{
                dal::detail::error_messages::
                    element_count_in_data_type_and_feature_type_arrays_does_not_match()
            };
        }
    }

    std::int64_t get_feature_count() const override {
        return dtypes_.get_count();
    }

    const dal::array<data_type>& get_data_types() const override {
        return dtypes_;
    }

    const dal::array<feature_type>& get_feature_types() const override {
        return ftypes_;
    }

    const feature_type& get_feature_type(std::int64_t i) const override {
        if (!is_in_range(i)) {
            throw out_of_range(dal::detail::error_messages::feature_index_is_out_of_range());
        }
        return ftypes_[i];
    }

    const data_type& get_data_type(std::int64_t i) const override {
        if (!is_in_range(i)) {
            throw out_of_range(dal::detail::error_messages::feature_index_is_out_of_range());
        }
        return dtypes_[i];
    }

    alloc_kind get_alloc_kind() const override {
        return kind_;
    }

    void serialize(detail::output_archive& ar) const override {
        ar(dtypes_, ftypes_);
    }

    void deserialize(detail::input_archive& ar) override {
        ar(dtypes_, ftypes_);
    }

private:
    bool is_in_range(std::int64_t i) const {
        return i >= 0 && i < dtypes_.get_count();
    }

    dal::array<data_type> dtypes_;
    dal::array<feature_type> ftypes_;
    alloc_kind kind_ = alloc_kind::non_usm; // Default allocation kind is non-USM
};

__ONEDAL_REGISTER_SERIALIZABLE__(simple_metadata_impl)

table_metadata::table_metadata() : impl_(new empty_metadata_impl()) {}

table_metadata::table_metadata(const dal::array<data_type>& dtypes,
                               const dal::array<feature_type>& ftypes)
        : impl_(new simple_metadata_impl(dtypes, ftypes, alloc_kind::non_usm)) {}

table_metadata::table_metadata(const dal::array<data_type>& dtypes,
                               const dal::array<feature_type>& ftypes,
                               const alloc_kind kind)
        : impl_(new simple_metadata_impl(dtypes, ftypes, kind)) {}

int64_t table_metadata::get_feature_count() const {
    return impl_->get_feature_count();
}

const feature_type& table_metadata::get_feature_type(std::int64_t feature_index) const {
    return impl_->get_feature_type(feature_index);
}

const data_type& table_metadata::get_data_type(std::int64_t feature_index) const {
    return impl_->get_data_type(feature_index);
}

const dal::array<feature_type>& table_metadata::get_feature_types() const {
    return impl_->get_feature_types();
}

const dal::array<data_type>& table_metadata::get_data_types() const {
    return impl_->get_data_types();
}

alloc_kind table_metadata::get_alloc_kind() const {
    return impl_->get_alloc_kind();
}

void table_metadata::serialize(detail::output_archive& ar) const {
    detail::serialize_polymorphic_shared(impl_, ar);
}

void table_metadata::deserialize(detail::input_archive& ar) {
    impl_ = detail::deserialize_polymorphic_shared<detail::table_metadata_impl>(ar);
}

table::table() : table(new backend::empty_table_impl{}) {}

table::table(table&& t) : impl_(std::move(t.impl_)) {
    t.impl_.reset(new backend::empty_table_impl{});
}

table& table::operator=(table&& t) {
    this->impl_.swap(t.impl_);
    return *this;
}

bool table::has_data() const noexcept {
    return impl_->get_column_count() > 0 && impl_->get_row_count() > 0;
}

int64_t table::get_column_count() const {
    return impl_->get_column_count();
}

int64_t table::get_row_count() const {
    return impl_->get_row_count();
}

const table_metadata& table::get_metadata() const {
    return impl_->get_metadata();
}

int64_t table::get_kind() const {
    return impl_->get_kind();
}

data_layout table::get_data_layout() const {
    return impl_->get_data_layout();
}

void table::serialize(detail::output_archive& ar) const {
    detail::serialize_polymorphic_shared(impl_, ar);
}

void table::deserialize(detail::input_archive& ar) {
    detail::deserialize_polymorphic_shared(impl_, ar);
}

void table::validate_input_dimensions(std::int64_t row_count, std::int64_t column_count) {
    if (row_count <= 0) {
        throw domain_error{ detail::error_messages::rc_leq_zero() };
    }

    if (column_count <= 0) {
        throw domain_error{ detail::error_messages::cc_leq_zero() };
    }
}

} // namespace v1
} // namespace oneapi::dal

// We need to make sure that all table types are registered for serialization,
// see "table/backend/register_serializable.cpp"
ONEDAL_FORCE_SERIALIZABLE_INIT(tables)
