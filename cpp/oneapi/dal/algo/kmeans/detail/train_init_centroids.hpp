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

#include <daal/include/data_management/data/numeric_table.h>

#include "oneapi/dal/algo/kmeans/common.hpp"

namespace oneapi::dal::kmeans::detail {

template <typename Float, typename Method, typename Table>
daal::data_management::NumericTablePtr daal_generate_centroids(
    const detail::descriptor_base<task::clustering>& desc,
    const Table& data);

} // namespace oneapi::dal::kmeans::detail
