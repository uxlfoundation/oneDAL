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

#include <daal/include/algorithms/multi_class_classifier/multi_class_classifier_model.h>

#include "oneapi/dal/backend/serialization.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/archive.hpp"

namespace oneapi::dal::svm::backend {

namespace daal_multiclass = daal::algorithms::multi_class_classifier;

class model_interop : public base {
public:
    virtual ~model_interop() = default;
};

template <typename DaalModel>
class model_interop_impl : public model_interop,
                           public ONEDAL_SERIALIZABLE(svm_model_interop_impl_multiclass_id) {
    using model_ptr_t = daal::services::SharedPtr<DaalModel>;

public:
    model_interop_impl() = default;

    model_interop_impl(const model_ptr_t& model) : daal_model_(model) {}

    const model_ptr_t get_model() const {
        return daal_model_;
    }

    void serialize(dal::detail::output_archive& ar) const override {
        dal::backend::interop::daal_output_data_archive daal_ar(ar);
        daal_ar.setSharedPtrObj(const_cast<model_ptr_t&>(daal_model_));
    }

    void deserialize(dal::detail::input_archive& ar) override {
        dal::backend::interop::daal_input_data_archive daal_ar(ar);
        daal_ar.setSharedPtrObj(daal_model_);
    }

private:
    model_ptr_t daal_model_;
};

using model_interop_cls = model_interop_impl<daal_multiclass::Model>;

} // namespace oneapi::dal::svm::backend
