/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <exception>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::preview::spmd {
namespace v1 {

class ONEDAL_EXPORT error_holder : public runtime_error, public std::runtime_error {
public:
    error_holder() = delete;
    explicit error_holder(const std::exception_ptr& actual);
    const char* what() const noexcept override;
    std::exception_ptr get_actual() const;
    void rethrow_actual() const;

private:
    std::exception_ptr actual_exception_;
};

class ONEDAL_EXPORT coworker_error : public runtime_error, public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    const char* what() const noexcept override;
};

class ONEDAL_EXPORT communication_error : public runtime_error, public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    const char* what() const noexcept override;
};

} // namespace v1

using v1::error_holder;
using v1::coworker_error;
using v1::communication_error;

} // namespace oneapi::dal::preview::spmd
