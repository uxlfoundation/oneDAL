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

#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_impl.hpp"

namespace oneapi::dal::backend::primitives {

template class kselect_by_rows<float>;
template std::int64_t kselect_by_rows_scratch_size<float>(const sycl::queue&,
                                                          const ndshape<2>&,
                                                          std::int64_t);

} // namespace oneapi::dal::backend::primitives
