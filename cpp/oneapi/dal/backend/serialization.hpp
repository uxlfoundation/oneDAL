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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/serialization.hpp"

#define ONEDAL_REGISTER_SERIALIZABLE __ONEDAL_REGISTER_SERIALIZABLE__

#define ONEDAL_REGISTER_SERIALIZABLE_INIT(unique_id) \
    namespace oneapi::dal::backend {                 \
    bool __force_serializable_init__##unique_id() {  \
        return true;                                 \
    }                                                \
    }

#define ONEDAL_FORCE_SERIALIZABLE_INIT(unique_id)                                        \
    namespace oneapi::dal::backend {                                                     \
    bool __force_serializable_init__##unique_id();                                       \
    }                                                                                    \
    [[maybe_unused]] static volatile bool __force_serializable_init_dummy__##unique_id = \
        ::oneapi::dal::backend::__force_serializable_init__##unique_id();

#define ONEDAL_SERIALIZATION_ID(id) ::oneapi::dal::backend::serialization_ids::id

#define ONEDAL_SERIALIZATION_ID_MAP(Tag, id) \
    ::oneapi::dal::backend::serialization_id_map<Tag, ONEDAL_SERIALIZATION_ID(id)>

#define ONEDAL_SERIALIZABLE(id) ::oneapi::dal::detail::serializable<ONEDAL_SERIALIZATION_ID(id)>

#define ONEDAL_SERIALIZABLE_MAP2(Tag, Map1, Map2) \
    ::oneapi::dal::backend::                      \
        serializable_map<Tag, ONEDAL_SERIALIZATION_ID_MAP Map1, ONEDAL_SERIALIZATION_ID_MAP Map2>

#define ONEDAL_SERIALIZABLE_MAP3(Tag, Map1, Map2, Map3)                        \
    ::oneapi::dal::backend::serializable_map<Tag,                              \
                                             ONEDAL_SERIALIZATION_ID_MAP Map1, \
                                             ONEDAL_SERIALIZATION_ID_MAP Map2, \
                                             ONEDAL_SERIALIZATION_ID_MAP Map3>

#define ONEDAL_SERIALIZABLE_MAP4(Tag, Map1, Map2, Map3, Map4)                  \
    ::oneapi::dal::backend::serializable_map<Tag,                              \
                                             ONEDAL_SERIALIZATION_ID_MAP Map1, \
                                             ONEDAL_SERIALIZATION_ID_MAP Map2, \
                                             ONEDAL_SERIALIZATION_ID_MAP Map3, \
                                             ONEDAL_SERIALIZATION_ID_MAP Map4>

namespace oneapi::dal::backend {

template <typename Tag, std::uint64_t SerializationId>
struct serialization_id_map {
    using tag_t = Tag;
    static constexpr std::uint64_t serialization_id_v = SerializationId;
};

template <typename... Args>
class serializable_map;

template <typename Tag, typename Map1, typename Map2>
class serializable_map<Tag, Map1, Map2> : public base, public dal::detail::serializable_iface {
public:
    static std::uint64_t serialization_id() {
        using tag1_t = typename Map1::tag_t;
        using tag2_t = typename Map2::tag_t;
        if constexpr (std::is_same_v<Tag, tag1_t>) {
            return Map1::serialization_id_v;
        }
        else if constexpr (std::is_same_v<Tag, tag2_t>) {
            return Map2::serialization_id_v;
        }
        ONEDAL_ASSERT(!"Unreachable");
        return 0;
    }

    std::uint64_t get_serialization_id() const override {
        return serialization_id();
    }
};

template <typename Tag, typename Map1, typename Map2, typename Map3>
class serializable_map<Tag, Map1, Map2, Map3> : public base,
                                                public dal::detail::serializable_iface {
public:
    static std::uint64_t serialization_id() {
        using tag1_t = typename Map1::tag_t;
        using tag2_t = typename Map2::tag_t;
        using tag3_t = typename Map3::tag_t;
        if constexpr (std::is_same_v<Tag, tag1_t>) {
            return Map1::serialization_id_v;
        }
        else if constexpr (std::is_same_v<Tag, tag2_t>) {
            return Map2::serialization_id_v;
        }
        else if constexpr (std::is_same_v<Tag, tag3_t>) {
            return Map3::serialization_id_v;
        }
        ONEDAL_ASSERT(!"Unreachable");
        return 0;
    }

    std::uint64_t get_serialization_id() const override {
        return serialization_id();
    }
};

template <typename Tag, typename Map1, typename Map2, typename Map3, typename Map4>
class serializable_map<Tag, Map1, Map2, Map3, Map4> : public base,
                                                      public dal::detail::serializable_iface {
public:
    static std::uint64_t serialization_id() {
        using tag1_t = typename Map1::tag_t;
        using tag2_t = typename Map2::tag_t;
        using tag3_t = typename Map3::tag_t;
        using tag4_t = typename Map4::tag_t;
        if constexpr (std::is_same_v<Tag, tag1_t>) {
            return Map1::serialization_id_v;
        }
        else if constexpr (std::is_same_v<Tag, tag2_t>) {
            return Map2::serialization_id_v;
        }
        else if constexpr (std::is_same_v<Tag, tag3_t>) {
            return Map3::serialization_id_v;
        }
        else if constexpr (std::is_same_v<Tag, tag4_t>) {
            return Map4::serialization_id_v;
        }
        ONEDAL_ASSERT(!"Unreachable");
        return 0;
    }

    std::uint64_t get_serialization_id() const override {
        return serialization_id();
    }
};

#define ID(unique_id, name) static constexpr std::uint64_t name = unique_id

class serialization_ids {
public:
    // Common
    ID(1000000000, array_id);

    // Tables
    ID(2000000000, empty_table_metadata_id);
    ID(2010000000, simple_table_metadata_id);
    ID(2020000000, empty_table_id);
    ID(2030000000, homogen_table_id);
    ID(2040000000, csr_table_id);
    ID(2050000000, heterogen_table_id);

    // Algorithms - SVM
    // v1 classification/nu_classification IDs were retired when the model
    // gained per-class support-vector counts; bumped to fail-fast on old
    // archives rather than silently mis-parsing them.
    ID(3010000000, svm_classification_model_impl_id);
    ID(3010100000, svm_regression_model_impl_id);
    ID(3010200000, svm_model_interop_impl_multiclass_id);
    ID(3010300000, svm_nu_classification_model_impl_id);
    ID(3010400000, svm_nu_regression_model_impl_id);

    // Algorithms - PCA
    ID(4010000000, pca_dim_reduction_model_impl_id);

    // Algorithms - KNN
    ID(5010000000, knn_brute_force_classification_model_impl_id);
    ID(5010100000, knn_kd_tree_classification_model_impl_id);
    ID(5010200000, knn_model_interop_id);
    ID(5010300000, knn_brute_force_search_model_impl_id);
    ID(5010400000, knn_kd_tree_search_model_impl_id);
    ID(5010500000, knn_brute_force_regression_model_impl_id);
    ID(5010600000, knn_kd_tree_regression_model_impl_id);

    // Algorithms - Decision Forest
    ID(6010000000, decision_forest_classification_model_impl_id);
    ID(6020000000, decision_forest_regression_model_impl_id);
    ID(6030000000, decision_forest_model_interop_impl_cls_id);
    ID(6040000000, decision_forest_model_interop_impl_reg_id);

    // Algorithms - Linear Regression
    ID(7010000000, linear_regression_model_impl_id);

    // Algorithms - KMeans
    ID(8010000000, kmeans_clustering_model_impl_id);

    // Algorithms - Logistic Regression
    ID(9010000000, logistic_regression_model_impl_id);
};

#undef ID

} // namespace oneapi::dal::backend
