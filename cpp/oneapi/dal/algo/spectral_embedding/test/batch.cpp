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

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/algo/spectral_embedding/test/fixture.hpp"

namespace oneapi::dal::spectral_embedding::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace sp_emb = oneapi::dal::spectral_embedding;

template <typename TestType>
class spectral_embedding_batch_test
        : public spectral_embedding_test<TestType, spectral_embedding_batch_test<TestType>> {
public:
    using base_t = spectral_embedding_test<TestType, spectral_embedding_batch_test<TestType>>;

    void gen_dimensions() {
        this->n_ = GENERATE(8);
        this->p_ = GENERATE(3);
    }
};

TEMPLATE_LIST_TEST_M(spectral_embedding_batch_test,
                     "spectral_embedding gold test",
                     "[spectral embedding][integration][cpu]",
                     spectral_embedding_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_gpu());

    this->test_gold_input();
}

} // namespace oneapi::dal::spectral_embedding::test
