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

#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include <iostream>
#include "oneapi/dal/algo/decision_forest/backend/gpu/infer_kernel_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace be = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;
using address = sycl::access::address_space;

using sycl::ext::oneapi::plus;
using sycl::ext::oneapi::minimum;
using sycl::ext::oneapi::maximum;

template <typename Float, typename Index, typename Task>
void infer_kernel_impl<Float, Index, Task>::validate_input(const descriptor_t& desc,
                                                           const model_t& model,
                                                           const table& data) const {
    if (data.get_row_count() == 0) {
        throw domain_error(msg::invalid_range_of_rows());
    }

    if (data.get_row_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_rows());
    }
    if (data.get_column_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_columns());
    }
    if (model.get_tree_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_number_of_trees());
    }

    if constexpr (std::is_same_v<Task, task::classification>) {
        if (desc.get_class_count() > de::limits<Index>::max()) {
            throw domain_error(msg::invalid_number_of_classes());
        }
    }
}

template <typename Float, typename Index, typename Task>
void infer_kernel_impl<Float, Index, Task>::init_params(infer_context_t& ctx,
                                                        const descriptor_t& desc,
                                                        const model_t& model,
                                                        const table& data) const {
    if constexpr (std::is_same_v<Task, task::classification>) {
        ctx.class_count = de::integral_cast<Index>(desc.get_class_count());
        ctx.voting_mode = desc.get_voting_mode();
    }
    ctx.row_count = de::integral_cast<Index>(data.get_row_count());
    ctx.column_count = de::integral_cast<Index>(data.get_column_count());

    ctx.tree_count = de::integral_cast<Index>(model.get_tree_count());

    ctx.tree_in_group_count = ctx.tree_in_group_count_min;

    if (ctx.tree_count > ctx.tree_count_large) {
        ctx.tree_in_group_count = ctx.tree_in_group_count_for_large;
    }
    else if (ctx.tree_count > ctx.tree_count_medium) {
        ctx.tree_in_group_count = ctx.tree_in_group_count_for_medium;
    }
    else if (ctx.tree_count > ctx.tree_count_small) {
        ctx.tree_in_group_count = ctx.tree_in_group_count_for_small;
    }

    ctx.row_block_count = 1;
    if (ctx.row_count > ctx.row_count_large) {
        ctx.row_block_count = ctx.row_block_count_for_large;
    }
    else if (ctx.row_count > ctx.row_count_medium) {
        ctx.row_block_count = ctx.row_block_count_for_medium;
    }
}

template <typename Float, typename Index, typename Task>
std::tuple<pr::ndarray<Float, 1>, sycl::event>
infer_kernel_impl<Float, Index, Task>::predict_by_tree_group_weighted(
    const infer_context_t& ctx,
    const pr::ndview<Float, 2>& data,
    const model_manager_t& mng,
    const be::event_vector& deps) {
    ONEDAL_PROFILER_TASK(predict_by_tree_group_weighted, queue_);

    const Index max_tree_size = mng.get_max_tree_size();

    auto [ftr_idx_list, lch_idx_or_class_id_list, ftr_value_list] = mng.get_serialized_data();
    auto class_proba_list = mng.get_class_probabilities_list();

    ONEDAL_ASSERT(data.get_count() == ctx.row_count * ctx.column_count);
    ONEDAL_ASSERT(ftr_idx_list.get_count() == ctx.tree_count * max_tree_size);
    ONEDAL_ASSERT(lch_idx_or_class_id_list.get_count() == ctx.tree_count * max_tree_size);
    ONEDAL_ASSERT(ftr_value_list.get_count() == ctx.tree_count * max_tree_size);
    ONEDAL_ASSERT(class_proba_list.get_count() == ctx.tree_count * ctx.class_count * max_tree_size);

    const Index row_count = ctx.row_count;
    const Index column_count = ctx.column_count;
    const Index class_count = ctx.class_count;
    const Index tree_count = ctx.tree_count;
    const Float scale = Float(1) / tree_count;

    const Float* data_ptr = data.get_data();
    const Index* ftr_idx_list_ptr = ftr_idx_list.get_data();
    const Index* lch_cls_list_ptr = lch_idx_or_class_id_list.get_data();
    const Float* ftr_val_list_ptr = ftr_value_list.get_data();
    const Float* cls_prb_list_ptr = class_proba_list.get_data();

    Index obs_tree_group_response_count = ctx.class_count * ctx.tree_in_group_count;
    std::cout << "first overflow here" << std::endl;
    de::check_mul_overflow(ctx.row_count, obs_tree_group_response_count);
    auto [obs_response_list, zero_obs_response_event] =
        pr::ndarray<Float, 1>::zeros(queue_,
                                     ctx.row_count * obs_tree_group_response_count,
                                     alloc::device);

    Float* obs_cls_hist_list_ptr = obs_response_list.get_mutable_data();

    auto local_size = ctx.max_local_size;
    const sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ ctx.row_block_count * local_size, ctx.tree_in_group_count },
                                      { local_size, 1 });

    sycl::event last_event = zero_obs_response_event;
    for (Index proc_tree_count = 0; proc_tree_count < tree_count;
         proc_tree_count += ctx.tree_in_group_count) {
        last_event = queue_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.depends_on(last_event);
            cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
                const Index local_id = item.get_local_id()[0];
                const Index local_size = item.get_local_range()[0];

                const Index n_groups = item.get_group_range(0);
                const Index group_id = item.get_group().get_group_id(0);
                const Index n_tree_groups = item.get_group_range(1);
                const Index tree_group_id = item.get_group().get_group_id(1);
                const Index tree_id = proc_tree_count + tree_group_id;
                const Index leaf_mark = impl_const_t::leaf_mark;

                const Index elem_count = row_count / n_groups + bool(row_count % n_groups);

                const Index ind_start = group_id * elem_count;
                const Index ind_end =
                    sycl::min(static_cast<Index>((group_id + 1) * elem_count), row_count);

                if (tree_id < tree_count) {
                    const Index* tree_ftr_idx = ftr_idx_list_ptr + tree_id * max_tree_size;
                    const Index* tree_lch_cls = lch_cls_list_ptr + tree_id * max_tree_size;
                    const Float* tree_ftr_val = ftr_val_list_ptr + tree_id * max_tree_size;
                    const Float* tree_cls_prb =
                        cls_prb_list_ptr + tree_id * max_tree_size * class_count;

                    bool tree_root_is_split = leaf_mark != tree_ftr_idx[0];

                    for (Index i = ind_start + local_id; i < ind_end; i += local_size) {
                        std::uint32_t tree_curr_node = 0;
                        bool node_is_split = tree_root_is_split;
                        for (; node_is_split > 0;) {
                            std::uint32_t idx =
                                static_cast<std::uint32_t>(tree_ftr_idx[tree_curr_node]);
                            std::uint32_t sn = static_cast<std::uint32_t>(
                                data_ptr[i * column_count + idx] > tree_ftr_val[tree_curr_node]);
                            tree_curr_node =
                                static_cast<std::uint32_t>(tree_lch_cls[tree_curr_node]) + sn;
                            node_is_split = leaf_mark != tree_ftr_idx[tree_curr_node];
                        }
                        for (Index class_idx = 0; class_idx < class_count; class_idx++) {
                            obs_cls_hist_list_ptr[i * n_tree_groups * class_count +
                                                  class_idx * n_tree_groups + tree_group_id] +=
                                scale * static_cast<Float>(
                                            tree_cls_prb[tree_curr_node * class_count + class_idx]);
                        }
                    }
                }
            });
        });
    }

    return std::make_tuple(obs_response_list, last_event);
}

template <typename Float, typename Index, typename Task>
std::tuple<pr::ndarray<Float, 1>, sycl::event>
infer_kernel_impl<Float, Index, Task>::predict_by_tree_group(const infer_context_t& ctx,
                                                             const pr::ndview<Float, 2>& data,
                                                             const model_manager_t& mng,
                                                             const be::event_vector& deps) {
    ONEDAL_PROFILER_TASK(predict_by_tree_group, queue_);

    constexpr bool is_classification = std::is_same_v<Task, task::classification>;
    const Index max_tree_size = mng.get_max_tree_size();

    auto [ftr_idx_list, lch_idx_or_class_id_list, ftr_value_list] = mng.get_serialized_data();

    ONEDAL_ASSERT(data.get_count() == ctx.row_count * ctx.column_count);
    ONEDAL_ASSERT(ftr_idx_list.get_count() == ctx.tree_count * max_tree_size);
    ONEDAL_ASSERT(lch_idx_or_class_id_list.get_count() == ctx.tree_count * max_tree_size);
    ONEDAL_ASSERT(ftr_value_list.get_count() == ctx.tree_count * max_tree_size);

    const Index row_count = ctx.row_count;
    const Index column_count = ctx.column_count;
    const Index class_count = ctx.class_count;
    const Index tree_count = ctx.tree_count;
    const Float scale = Float(1) / tree_count;

    const Float* data_ptr = data.get_data();
    const Index* ftr_idx_list_ptr = ftr_idx_list.get_data();
    const Index* lch_cls_list_ptr = lch_idx_or_class_id_list.get_data();
    const Float* ftr_val_list_ptr = ftr_value_list.get_data();

    Index obs_tree_group_response_count = ctx.tree_in_group_count;
    if constexpr (is_classification) {
        obs_tree_group_response_count = ctx.class_count * ctx.tree_in_group_count;
        std::cout << "second overflow here" << std::endl;
        de::check_mul_overflow(ctx.row_count, obs_tree_group_response_count);
    }
    auto [obs_response_list, zero_obs_response_event] =
        pr::ndarray<Float, 1>::zeros(queue_,
                                     ctx.row_count * obs_tree_group_response_count,
                                     alloc::device);

    Float* obs_cls_hist_list_ptr = obs_response_list.get_mutable_data();

    auto local_size = ctx.max_local_size;
    const sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ ctx.row_block_count * local_size, ctx.tree_in_group_count },
                                      { local_size, 1 });

    sycl::event last_event = zero_obs_response_event;
    for (Index proc_tree_count = 0; proc_tree_count < tree_count;
         proc_tree_count += ctx.tree_in_group_count) {
        last_event = queue_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.depends_on(last_event);
            cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
                const Index local_id = item.get_local_id()[0];
                const Index local_size = item.get_local_range()[0];

                const Index n_groups = item.get_group_range(0);
                const Index group_id = item.get_group().get_group_id(0);
                const Index n_tree_groups = item.get_group_range(1);
                const Index tree_group_id = item.get_group().get_group_id(1);
                const Index tree_id = proc_tree_count + tree_group_id;
                const Index leaf_mark = impl_const_t::leaf_mark;

                const Index elem_count = row_count / n_groups + bool(row_count % n_groups);

                const Index ind_start = group_id * elem_count;
                const Index ind_end =
                    sycl::min(static_cast<Index>((group_id + 1) * elem_count), row_count);

                if (tree_id < tree_count) {
                    const Index* tree_ftr_idx = ftr_idx_list_ptr + tree_id * max_tree_size;
                    const Index* tree_lch_cls = lch_cls_list_ptr + tree_id * max_tree_size;
                    const Float* tree_ftr_val = ftr_val_list_ptr + tree_id * max_tree_size;

                    bool tree_root_is_split = leaf_mark != tree_ftr_idx[0];

                    for (Index i = ind_start + local_id; i < ind_end; i += local_size) {
                        std::uint32_t tree_curr_node = 0;
                        bool node_is_split = tree_root_is_split;
                        for (; node_is_split > 0;) {
                            std::uint32_t idx = node_is_split * tree_ftr_idx[tree_curr_node];
                            std::uint32_t sn = static_cast<std::uint32_t>(
                                data_ptr[i * column_count + idx] > tree_ftr_val[tree_curr_node]);
                            tree_curr_node -=
                                node_is_split *
                                (tree_curr_node -
                                 static_cast<std::uint32_t>(tree_lch_cls[tree_curr_node]) - sn);
                            node_is_split = leaf_mark != tree_ftr_idx[tree_curr_node];
                        }

                        if constexpr (is_classification) {
                            Index class_idx = tree_lch_cls[tree_curr_node];
                            obs_cls_hist_list_ptr[i * n_tree_groups * class_count +
                                                  class_idx * n_tree_groups + tree_group_id] +=
                                scale;
                        }
                        else {
                            obs_cls_hist_list_ptr[i * n_tree_groups + tree_group_id] +=
                                tree_ftr_val[tree_curr_node];
                        }
                    }
                }
            });
        });
    }

    return std::make_tuple(obs_response_list, last_event);
}

template <typename Float, typename Index, typename Task>
std::tuple<pr::ndarray<Float, 1>, sycl::event>
infer_kernel_impl<Float, Index, Task>::reduce_tree_group_response(
    const infer_context_t& ctx,
    const pr::ndview<Float, 1>& obs_response_list,
    const be::event_vector& deps) {
    ONEDAL_PROFILER_TASK(reduce_tree_group_response, queue_);

    constexpr bool is_classification = std::is_same_v<Task, task::classification>;

    Index response_count = ctx.row_count;

    if constexpr (is_classification) {
        ONEDAL_ASSERT(obs_response_list.get_count() ==
                      ctx.row_count * ctx.class_count * ctx.tree_in_group_count);
        std::cout << "third overflow here" << std::endl;
        de::check_mul_overflow(ctx.class_count, ctx.row_count);
        response_count = ctx.row_count * ctx.class_count;
    }
    else {
        ONEDAL_ASSERT(obs_response_list.get_count() == ctx.row_count * ctx.tree_in_group_count);
    }

    auto [response_list, zero_response_event] =
        pr::ndarray<Float, 1>::zeros(queue_, response_count, alloc::device);

    const Index class_count = ctx.class_count;
    const Index row_count = ctx.row_count;
    const Index tree_in_group_count = ctx.tree_in_group_count;
    const Index tree_count = ctx.tree_count;
    const Float scale = Float(1) / tree_count;

    const Float* obs_response_list_ptr = obs_response_list.get_data();
    Float* response_list_ptr = response_list.get_mutable_data();

    const auto local_size = be::device_max_sg_size(queue_);

    const sycl::nd_range<1> nd_range =
        be::make_multiple_nd_range_1d({ ctx.max_group_count * local_size }, { local_size });

    sycl::event last_event = zero_response_event;
    last_event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.depends_on(last_event);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }
            const Index group_id = item.get_group().get_group_id(0);
            const Index n_groups = item.get_group_range(0);
            const Index sub_group_local_id = sbg.get_local_id();
            const Index sub_group_size = sbg.get_local_range()[0];

            const Index elem_count = row_count / n_groups + bool(row_count % n_groups);

            const Index ind_start = group_id * elem_count;
            const Index ind_end =
                sycl::min(static_cast<Index>((group_id + 1) * elem_count), row_count);

            // obs_response_list_ptr each row contains certain class values from each tree for this observation
            // obs_response_list_ptr[0] = obs0_cls0_val_from_tree0, obs0_cls0_val_from_tree1 ... obs0_cls1_val_from_tree0, obs0_cls1_val_from_tree1 ...
            // obs_response_list_ptr[1] = obs1_cls0_val_from_tree0, obs1_cls0_val_from_tree1 ... obs1_cls1_val_from_tree0, obs1_cls1_val_from_tree1 ...

            for (Index row_idx = ind_start; row_idx < ind_end; row_idx++) {
                if constexpr (is_classification) {
                    for (Index class_idx = 0; class_idx < class_count; class_idx++) {
                        Index resp_offset = row_idx * tree_in_group_count * class_count +
                                            class_idx * tree_in_group_count;

                        Float resp_val = Float(0);
                        for (Index i = sub_group_local_id; i < tree_in_group_count;
                             i += sub_group_size) {
                            resp_val += obs_response_list_ptr[resp_offset + i];
                        }

                        resp_val = sycl::reduce_over_group(sbg, resp_val, plus<Float>());

                        if (0 == sub_group_local_id) {
                            response_list_ptr[row_idx * class_count + class_idx] = resp_val;
                        }
                    }
                }
                else {
                    Index resp_offset = row_idx * tree_in_group_count;

                    Float resp_val = Float(0);
                    for (Index i = sub_group_local_id; i < tree_in_group_count;
                         i += sub_group_size) {
                        resp_val += obs_response_list_ptr[resp_offset + i];
                    }

                    resp_val = sycl::reduce_over_group(sbg, resp_val, plus<Float>());

                    if (0 == sub_group_local_id) {
                        response_list_ptr[row_idx] = resp_val * scale;
                    }
                }
            }
        });
    });

    return std::make_tuple(response_list, last_event);
}

template <typename Float, typename Index, typename Task>
std::tuple<pr::ndarray<Float, 1>, sycl::event>
infer_kernel_impl<Float, Index, Task>::determine_winner(const infer_context_t& ctx,
                                                        const pr::ndview<Float, 1>& response_list,
                                                        const be::event_vector& deps) {
    ONEDAL_PROFILER_TASK(determine_winner, queue_);

    ONEDAL_ASSERT(response_list.get_count() == ctx.row_count * ctx.class_count);
    auto winner_list = pr::ndarray<Float, 1>::empty(queue_, { ctx.row_count }, alloc::device);

    Index class_count = ctx.class_count;
    Index row_count = ctx.row_count;
    const Float* response_list_ptr = response_list.get_data();
    Float* winner_list_ptr = winner_list.get_mutable_data();

    const sycl::nd_range<1> nd_range =
        be::make_multiple_nd_range_1d({ ctx.max_group_count * ctx.max_local_size },
                                      { ctx.max_local_size });

    sycl::event last_event;
    last_event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            const Index local_id = item.get_local_id()[0];
            const Index local_size = item.get_local_range()[0];
            const Index n_groups = item.get_group_range(0);
            const Index group_id = item.get_group().get_group_id(0);

            const Index elem_count = row_count / n_groups + bool(row_count % n_groups);

            const Index ind_start = group_id * elem_count;
            const Index ind_end =
                sycl::min(static_cast<Index>((group_id + 1) * elem_count), row_count);

            for (Index i = ind_start + local_id; i < ind_end; i += local_size) {
                Float class_count_max = (Float)0;
                Float class_winner = (Float)0;
                for (Index class_idx = 0; class_idx < class_count; class_idx++) {
                    if (class_count_max < response_list_ptr[i * class_count + class_idx]) {
                        class_count_max = response_list_ptr[i * class_count + class_idx];
                        class_winner = Float(class_idx);
                    }
                }
                winner_list_ptr[i] = class_winner;
            }
        });
    });

    return std::make_tuple(winner_list, last_event);
}

template <typename Float, typename Index, typename Task>
infer_result<Task> infer_kernel_impl<Float, Index, Task>::operator()(const descriptor_t& desc,
                                                                     const model_t& model,
                                                                     const table& data) {
    validate_input(desc, model, data);

    infer_context_t ctx;
    init_params(ctx, desc, model, data);
    model_manager_t model_mng(queue_, ctx, model);

    result_t res;

    const auto data_nd = pr::table2ndarray<Float>(queue_, data, alloc::device);

    pr::ndarray<Float, 1> tree_group_response_list;
    pr::ndarray<Float, 1> response_list;
    sycl::event predict_event;

    if constexpr (std::is_same_v<Task, task::classification>) {
        if (voting_mode::weighted == ctx.voting_mode && model_mng.is_weighted_available()) {
            std::tie(tree_group_response_list, predict_event) =
                predict_by_tree_group_weighted(ctx, data_nd, model_mng);
            std::tie(response_list, predict_event) =
                reduce_tree_group_response(ctx, tree_group_response_list, { predict_event });
        }
        else {
            std::tie(tree_group_response_list, predict_event) =
                predict_by_tree_group(ctx, data_nd, model_mng);
            std::tie(response_list, predict_event) =
                reduce_tree_group_response(ctx, tree_group_response_list, { predict_event });
        }
    }
    else {
        std::tie(tree_group_response_list, predict_event) =
            predict_by_tree_group(ctx, data_nd, model_mng);
        std::tie(response_list, predict_event) =
            reduce_tree_group_response(ctx, tree_group_response_list, { predict_event });
    }

    if constexpr (std::is_same_v<Task, task::classification>) {
        auto [response, winner_event] = determine_winner(ctx, response_list, { predict_event });

        if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_responses)) {
            res.set_responses(
                homogen_table::wrap(response.flatten(queue_, { winner_event }), ctx.row_count, 1));
        }

        if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_probabilities)) {
            res.set_probabilities(
                homogen_table::wrap(response_list.flatten(queue_, { predict_event }),
                                    ctx.row_count,
                                    ctx.class_count));
        }
    }
    else {
        res.set_responses(homogen_table::wrap(response_list.flatten(queue_, { predict_event }),
                                              ctx.row_count,
                                              1));
    }

    if (comm_.get_rank_count() > 1) {
        comm_.wait_for_exception_handling();
    }

    return res;
}

#define INSTANTIATE(F, I, T) template class infer_kernel_impl<F, I, T>;

INSTANTIATE(float, std::int32_t, task::classification);
INSTANTIATE(float, std::int32_t, task::regression);

INSTANTIATE(double, std::int32_t, task::classification);
INSTANTIATE(double, std::int32_t, task::regression);

} // namespace oneapi::dal::decision_forest::backend
