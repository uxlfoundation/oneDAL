/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/table/common.hpp"

#include <mkl_version.h>
#include <oneapi/mkl.hpp>

namespace oneapi::dal::backend::primitives {

namespace mkl = oneapi::mkl;

/// Convert oneDAL `sparse_indexing` to oneMKL `index_base`
inline constexpr mkl::index_base sparse_indexing_to_mkl(const sparse_indexing indexing) {
    return (indexing == sparse_indexing::zero_based) ? mkl::index_base::zero : mkl::index_base::one;
}

} // namespace oneapi::dal::backend::primitives
