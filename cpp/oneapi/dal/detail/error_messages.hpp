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

#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::detail {
namespace v1 {

#define MSG(id) static const char* id() noexcept

// -----------------------------------
// Guide for adding new error messages
// -----------------------------------
// - Before adding a new message check if there is a similar message
//   that can be reused. If you see some algorithm-specific message
//   can be reused in other algorithm/module, consider moving this to
//   common groups like `Common` or `General Algorithms`.
//
// - Select appropriate group among `Common`, `Tables`, ...
//   for a new message.
//
// - Keep alphabtical order of messages within each group.
//   If a new algorithm-specific group is needed, add it
//   in alphabetical order
//
// - Use LaTeX-like shortening for mathematical operations,
//   for example:
//
//   =    eq
//   !=   neq
//   >    gt
//   >=   geq
//   <    lt
//   <=   leq
//
// - If a new mathematical operator introduced, stick to LaTeX
//   conventions as well.
//
// - For Intel(R) oneAPI Data Analytics Library (oneDAL) specific terms use the following shortening:
//
//   row count      rc
//   column count   cc
//   descriptor     desc

class ONEDAL_EXPORT error_messages {
public:
    error_messages() = delete;

    /* Common */
    MSG(array_does_not_contain_mutable_data);
    MSG(algorithm_is_not_implemented_for_this_device);
    MSG(feature_index_is_out_of_range);
    MSG(incompatible_array_reinterpret_cast_types);
    MSG(integral_type_conversion_overflow);
    MSG(integral_type_conversion_underflow);
    MSG(invalid_data_block_size);
    MSG(invalid_column_indices_block_size);
    MSG(method_not_implemented);
    MSG(negative_integral_value_conversion_to_unsigned);
    MSG(only_homogen_table_is_supported);
    MSG(overflow_found_in_multiplication_of_two_values);
    MSG(overflow_found_in_sum_of_two_values);
    MSG(queues_in_different_contexts);
    MSG(small_data_block);
    MSG(spmd_version_of_algorithm_is_not_implemented);
    MSG(spmd_version_of_algorithm_is_not_implemented_for_this_device);
    MSG(unknown_memcpy_error);
    MSG(unknown_status_code);
    MSG(unknown_usm_pointer_type);
    MSG(unsupported_data_layout);
    MSG(unsupported_data_type);
    MSG(unsupported_device_type);
    MSG(unsupported_feature_type);
    MSG(unsupported_usm_alloc);
    MSG(page_size_leq_zero);
    MSG(invalid_key);
    MSG(capacity_leq_zero);
    MSG(empty_set_of_result_options);
    MSG(this_result_is_not_enabled_via_result_options);
    MSG(spmd_error_holder_message);
    MSG(spmd_coworker_failure);
    MSG(unexpected_dataset_type);

    /* Primitives */
    MSG(invalid_number_of_elements_to_process);
    MSG(invalid_number_of_elements_to_sort);
    MSG(failed_to_compute_eigenvectors);
    MSG(failed_to_generate_random_numbers);

    /* Tables */
    MSG(allocated_memory_size_is_not_enough_to_copy_data);
    MSG(cannot_get_data_type_from_empty_metadata);
    MSG(cannot_get_feature_type_from_empty_metadata);
    MSG(cc_leq_zero);
    MSG(element_count_in_data_type_and_feature_type_arrays_does_not_match);
    MSG(pulling_column_is_not_supported_for_dpc);
    MSG(pulling_column_is_not_supported);
    MSG(pulling_rows_is_not_supported_for_dpc);
    MSG(pulling_rows_is_not_supported);
    MSG(pushing_column_is_not_supported_for_dpc);
    MSG(pushing_column_is_not_supported);
    MSG(pushing_rows_is_not_supported_for_dpc);
    MSG(pushing_rows_is_not_supported);
    MSG(rc_and_cc_do_not_match_element_count_in_array);
    MSG(rc_leq_zero);
    MSG(object_does_not_provide_read_access_to_rows);
    MSG(object_does_not_provide_write_access_to_rows);
    MSG(object_does_not_provide_read_access_to_columns);
    MSG(object_does_not_provide_write_access_to_columns);
    MSG(object_does_not_provide_access_to_rows_or_columns);
    MSG(unsupported_conversion_types);
    MSG(invalid_first_row_offset);
    MSG(row_offsets_lt_min_value);
    MSG(row_offsets_gt_max_value);
    MSG(row_offsets_not_ascending);
    MSG(row_offsets_pointer_is_null);
    MSG(column_indices_lt_min_value);
    MSG(column_indices_gt_max_value);
    MSG(zero_based_indexing_is_not_supported);
    MSG(object_does_not_provide_read_access_to_csr);
    MSG(pull_column_interface_is_not_implemented);

    /* Ranges */
    MSG(invalid_range_of_rows);
    MSG(invalid_range_of_columns);
    MSG(column_index_out_of_range);

    /* RNG */
    MSG(rng_engine_does_not_support_parallelization_techniques);
    MSG(rng_engine_is_not_supported);

    /* Graphs */
    MSG(vertex_index_out_of_range_expect_from_zero_to_vertex_count);
    MSG(negative_vertex_id);
    MSG(unimplemented_sorting_procedure);
    MSG(edge_values_are_empty);

    /* I/O */
    MSG(file_not_found);
    MSG(unsupported_read_mode);

    /* Serialization */
    MSG(object_is_not_serializable);
    MSG(archive_content_does_not_match_type);
    MSG(archive_is_in_invalid_state);

    /* General Algorithms */
    MSG(accuracy_threshold_lt_zero);
    MSG(class_count_leq_one);
    MSG(conv_tol_lt_zero);
    MSG(input_data_is_empty);
    MSG(input_data_rc_neq_input_responses_rc);
    MSG(input_data_rc_neq_input_weights_rc);
    MSG(input_responses_are_empty);
    MSG(input_responses_contain_only_one_unique_value_expect_two);
    MSG(input_responses_contain_wrong_unique_values_count_expect_two);
    MSG(input_responses_table_has_wrong_cc_expect_one);
    MSG(iteration_count_lt_zero);
    MSG(max_iteration_count_leq_zero);
    MSG(max_iteration_count_lt_zero);

    /* Decision Forest */
    MSG(bootstrap_is_incompatible_with_error_metric);
    MSG(bootstrap_is_incompatible_with_variable_importance_mode);
    MSG(decision_forest_train_dense_method_is_not_implemented_for_gpu);
    MSG(decision_forest_train_hist_method_is_not_implemented_for_cpu);
    MSG(invalid_number_of_trees);
    MSG(invalid_number_of_classes);
    MSG(input_model_is_not_initialized);
    MSG(invalid_number_of_min_observations_in_leaf_node);
    MSG(invalid_number_of_feature_per_node);
    MSG(invalid_number_of_max_bins);
    MSG(invalid_value_for_min_bin_size);
    MSG(invalid_value_for_observations_per_tree_fraction);
    MSG(not_enough_memory_to_build_one_tree);
    MSG(not_enough_local_memory_for_hist);
    MSG(input_model_tree_has_invalid_size);

    /* Jaccard */
    MSG(column_begin_gt_column_end);
    MSG(empty_edge_list);
    MSG(interval_gt_vertex_count);
    MSG(negative_interval);
    MSG(row_begin_gt_row_end);
    MSG(range_idx_gt_max_int32);

    /* Subgraph Isomorphism */
    MSG(unsupported_kind);
    MSG(max_match_count_lt_zero);
    MSG(empty_target_graph);
    MSG(empty_pattern_graph);
    MSG(subgraph_isomorphism_is_not_implemented_for_labeled_edges);
    MSG(incorrect_index_is_returned);
    MSG(invalid_vertex_edge_attributes);
    MSG(target_graph_is_smaller_than_pattern_graph);

    /* K-Means and K-Means Init */
    MSG(cluster_count_leq_zero);
    MSG(cluster_count_exceeds_data_row_count);
    MSG(cluster_count_gt_max_int32);
    MSG(row_count_gt_max_int32);
    MSG(input_initial_centroids_are_empty);
    MSG(input_initial_centroids_cc_neq_input_data_cc);
    MSG(input_initial_centroids_rc_neq_desc_cluster_count);
    MSG(input_model_centroids_are_empty);
    MSG(input_model_centroids_cc_neq_input_data_cc);
    MSG(input_model_centroids_rc_neq_desc_cluster_count);
    MSG(kmeans_init_csr_methods_are_not_implemented_for_gpu);
    MSG(kmeans_init_parallel_plus_dense_method_is_not_implemented_for_gpu);
    MSG(kmeans_init_plus_plus_dense_method_is_not_implemented_for_gpu);
    MSG(objective_function_value_lt_zero);

    /* k-NN */
    MSG(knn_kd_tree_method_is_not_implemented_for_gpu);
    MSG(knn_regression_task_is_not_implemented_for_cpu);
    MSG(knn_search_task_is_not_implemented_for_gpu);
    MSG(neighbor_count_lt_one);
    MSG(unknown_distance_type);
    MSG(distance_is_not_supported_for_gpu);
    MSG(incompatible_knn_model);
    MSG(invalid_set_of_result_options_to_search);

    /* Linear and RBF Kernels */
    MSG(input_x_cc_neq_y_cc);
    MSG(input_x_is_empty);
    MSG(input_y_is_empty);

    /* Linear Regression */
    MSG(intercept_result_option_requires_intercept_flag);
    MSG(too_small_singular_values);

    /* Logistic Regression */
    MSG(class_count_neq_two);
    MSG(inverse_regularization_leq_zero);
    MSG(l1_coef_neq_zero);
    MSG(log_reg_dense_batch_method_is_not_implemented_for_cpu);
    MSG(log_reg_sparse_method_is_not_implemented_for_cpu);
    MSG(unknown_optimizer);

    /* Louvain */
    MSG(negative_resolution);
    MSG(input_initial_partition_table_rc_neq_vertex_count);
    MSG(input_initial_partition_table_has_wrong_cc_expect_one);
    MSG(negative_initial_partition_label);
    MSG(initial_partition_label_gte_vertex_count);

    /* Minkowski distance */
    MSG(invalid_minkowski_degree);

    /* Objective function */
    MSG(resp_column_count_is_not_eq_to_one);
    MSG(params_column_count_is_not_eq_to_one);
    MSG(value_is_not_provided);
    MSG(gradient_is_not_provided);
    MSG(hessian_is_not_provided);
    MSG(incorrect_output_table_size);
    MSG(regularization_coef_is_less_than_zero);
    MSG(regularization_coef_is_nan_or_inf);

    /* Optimizers */

    MSG(matrix_is_not_positively_definite);

    /* PCA */
    MSG(component_count_lt_zero);
    MSG(input_data_cc_lt_desc_component_count);
    MSG(input_model_eigenvectors_cc_neq_input_data_cc);
    MSG(input_model_eigenvectors_rc_neq_desc_component_count);
    MSG(input_model_eigenvectors_rc_neq_input_data_cc);
    MSG(pca_svd_based_method_is_not_implemented_for_gpu);

    /* Shortest Paths */
    MSG(negative_source);
    MSG(source_gte_vertex_count);
    MSG(negative_delta);
    MSG(nothing_to_compute);
    MSG(distances_are_uninitialized);
    MSG(predecessors_are_uninitialized);

    /* SVM */
    MSG(c_leq_zero);
    MSG(cache_size_lt_zero);
    MSG(degree_lt_zero);
    MSG(input_model_coeffs_are_empty);
    MSG(input_model_coeffs_rc_neq_input_model_support_vector_count);
    MSG(input_model_does_not_match_kernel_function);
    MSG(input_model_support_vectors_are_empty);
    MSG(input_model_support_vectors_cc_neq_input_data_cc);
    MSG(input_model_support_vectors_rc_neq_input_model_support_vector_count);
    MSG(nu_gt_one);
    MSG(nu_leq_zero);
    MSG(nu_svm_smo_method_is_not_implemented_for_gpu);
    MSG(nu_svm_thunder_method_is_not_implemented_for_gpu);
    MSG(polynomial_kernel_is_not_implemented_for_gpu);
    MSG(sigmoid_kernel_is_not_implemented_for_gpu);
    MSG(sigma_leq_zero);
    MSG(svm_multiclass_not_implemented_for_gpu);
    MSG(svm_nu_classification_task_is_not_implemented_for_gpu);
    MSG(svm_nu_regression_task_is_not_implemented_for_gpu);
    MSG(svm_regression_task_is_not_implemented_for_gpu);
    MSG(svm_smo_method_is_not_implemented_for_gpu);
    MSG(tau_leq_zero);
    MSG(epsilon_lt_zero);
    MSG(unknown_kernel_function_type);

    /* DBSCAN & Basic Statistics*/
    MSG(weight_dimension_doesnt_match_data_dimension);
    MSG(weights_column_count_ne_1);

    /*SPMD*/
    MSG(unsupported_communicator_backend);
    MSG(invalid_data_type);
    MSG(invalid_op);
    MSG(invalid_buffer);
    MSG(invalid_count);
    MSG(invalid_mpi_comm);
    MSG(invalid_root);
    MSG(unknown_mpi_error);
    MSG(sendrecv_replace_is_not_implemented_for_threaded_communicator);
};

#undef MSG

} // namespace v1

using v1::error_messages;

} // namespace oneapi::dal::detail
