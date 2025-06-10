/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <daal/src/services/service_defines.h>

#include "oneapi/dal/algo/jaccard/backend/cpu/vertex_similarity_default_kernel.hpp"
#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/algo/jaccard/detail/service.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/backend/primitives/intersection/intersection.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/service_functions_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::preview::jaccard::backend {

using namespace preview::backend;

template <typename Cpu>
vertex_similarity_result<task::all_vertex_pairs> jaccard_avx512(
    const detail::descriptor_base<task::all_vertex_pairs> &desc,
    const dal::preview::detail::topology<std::int32_t> &t,
    void *result_ptr) {
    const auto rows_vertex = t._rows_vertex.get_data();
    const auto cols = t._cols.get_data();
    const auto degrees = t._degrees.get_data();

    const auto row_begin = dal::detail::integral_cast<std::int32_t>(desc.get_row_range_begin());
    const auto row_end = dal::detail::integral_cast<std::int32_t>(desc.get_row_range_end());
    const auto column_begin =
        dal::detail::integral_cast<std::int32_t>(desc.get_column_range_begin());
    const auto column_end = dal::detail::integral_cast<std::int32_t>(desc.get_column_range_end());
    const auto number_elements_in_block =
        detail::compute_number_elements_in_block(row_begin, row_end, column_begin, column_end);
    std::int32_t *first_vertices = reinterpret_cast<std::int32_t *>(result_ptr);
    std::int32_t *second_vertices = first_vertices + number_elements_in_block;
    float *jaccard = reinterpret_cast<float *>(second_vertices + number_elements_in_block);

    std::int64_t nnz = 0;
    std::int32_t j = column_begin;
#if defined(__AVX512F__) && defined(DAAL_INTEL_CPP_COMPILER)
    __m512i j_vertices_tmp1 =
        _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    GRAPH_STACK_ALING(64) std::int32_t stack16_j_vertex[16] = { 0 };

    std::int32_t ones_num = 0;
#endif

    for (std::int32_t i = row_begin; i < row_end; ++i) {
        const auto i_neighbor_size = degrees[i];
        const auto i_neigbhors = cols + rows_vertex[i];
        const auto diagonal = detail::min(i, column_end);

#if defined(__AVX512F__) && defined(DAAL_INTEL_CPP_COMPILER)
        __m512i n_i_start_v = _mm512_set1_epi32(i_neigbhors[0]);
        __m512i n_i_end_v = _mm512_set1_epi32(i_neigbhors[i_neighbor_size - 1]);
        __m512i i_vertex = _mm512_set1_epi32(i);

        if (j < column_begin + ((diagonal - column_begin) / 16) * 16) {
            //load_data(0)
            __m512i start_indices_j_v = _mm512_load_epi32(rows_vertex + j);
            __m512i end_indices_j_v_tmp = _mm512_load_epi32(rows_vertex + j + 1);
            __m512i end_indices_j_v = _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));

            __m512i n_j_start_v =
                _mm512_permutexvar_epi32(j_vertices_tmp1,
                                         _mm512_i32gather_epi32(start_indices_j_v, cols, 4));
            __m512i n_j_end_v =
                _mm512_permutexvar_epi32(j_vertices_tmp1,
                                         _mm512_i32gather_epi32(end_indices_j_v, cols, 4));

            for (; j + 16 < column_begin + ((diagonal - column_begin) / 16) * 16;) {
                __m512i start_indices_j_v = _mm512_load_epi32(rows_vertex + j + 16);
                __m512i end_indices_j_v_tmp = _mm512_load_epi32(rows_vertex + j + 17);
                __m512i end_indices_j_v =
                    _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));

                __m512i n_j_start_v1 =
                    _mm512_permutexvar_epi32(j_vertices_tmp1,
                                             _mm512_i32gather_epi32(start_indices_j_v, cols, 4));
                __m512i n_j_end_v1 =
                    _mm512_permutexvar_epi32(j_vertices_tmp1,
                                             _mm512_i32gather_epi32(end_indices_j_v, cols, 4));

                __mmask16 cmpgt1 = _mm512_cmpgt_epi32_mask(n_i_start_v, n_j_end_v);
                __mmask16 cmpgt2 = _mm512_cmpgt_epi32_mask(n_j_start_v, n_i_end_v);

                __mmask16 worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
                ones_num = _popcnt32_redef(_cvtmask16_u32(worth_intersection));

                if (ones_num != 0) {
                    __m512i j_vertices_tmp2 = _mm512_set1_epi32(j);
                    __m512i j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                    _mm512_mask_compressstoreu_epi32((stack16_j_vertex),
                                                     worth_intersection,
                                                     j_vertices);

                    GRAPH_STACK_ALING(64) std::int32_t stack16_intersections[16] = { 0 };
                    for (std::int32_t s = 0; s < ones_num; s++) {
                        const auto j_neighbor_size = degrees[stack16_j_vertex[s]];
                        const auto j_neigbhors = cols + rows_vertex[stack16_j_vertex[s]];
                        stack16_intersections[s] =
                            preview::backend::intersection<Cpu>(i_neigbhors,
                                                                j_neigbhors,
                                                                i_neighbor_size,
                                                                j_neighbor_size);
                    }
                    __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                    j_vertices = _mm512_load_epi32(stack16_j_vertex);

                    __mmask16 non_zero_coefficients =
                        _mm512_test_epi32_mask(intersections_v, intersections_v);
                    _mm512_mask_compressstoreu_epi32((first_vertices + nnz),
                                                     non_zero_coefficients,
                                                     i_vertex);
                    _mm512_mask_compressstoreu_epi32((second_vertices + nnz),
                                                     non_zero_coefficients,
                                                     j_vertices);
                    __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                    _mm512_mask_compressstoreu_ps((jaccard + nnz), non_zero_coefficients, tmp_v);

                    nnz += _popcnt32_redef(_cvtmask16_u32(non_zero_coefficients));
                    ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
                }

                j += 16;

                n_j_start_v = n_j_start_v1;
                n_j_end_v = n_j_end_v1;
            }

            //process n data

            __mmask16 cmpgt1 = _mm512_cmpgt_epi32_mask(n_i_start_v, n_j_end_v);
            __mmask16 cmpgt2 = _mm512_cmpgt_epi32_mask(n_j_start_v, n_i_end_v);

            __mmask16 worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
            ones_num = _popcnt32_redef(_cvtmask16_u32(worth_intersection));

            if (ones_num != 0) {
                __m512i j_vertices_tmp2 = _mm512_set1_epi32(j);
                __m512i j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                _mm512_mask_compressstoreu_epi32((stack16_j_vertex),
                                                 worth_intersection,
                                                 j_vertices);

                GRAPH_STACK_ALING(64) std::int32_t stack16_intersections[16] = { 0 };
                for (std::int32_t s = 0; s < ones_num; s++) {
                    const auto j_neighbor_size = degrees[stack16_j_vertex[s]];
                    const auto j_neigbhors = cols + rows_vertex[stack16_j_vertex[s]];
                    stack16_intersections[s] = preview::backend::intersection<Cpu>(i_neigbhors,
                                                                                   j_neigbhors,
                                                                                   i_neighbor_size,
                                                                                   j_neighbor_size);
                }
                __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                j_vertices = _mm512_load_epi32(stack16_j_vertex);
                __mmask16 non_zero_coefficients =
                    _mm512_test_epi32_mask(intersections_v, intersections_v);
                _mm512_mask_compressstoreu_epi32((first_vertices + nnz),
                                                 non_zero_coefficients,
                                                 i_vertex);
                _mm512_mask_compressstoreu_epi32((second_vertices + nnz),
                                                 non_zero_coefficients,
                                                 j_vertices);
                __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                _mm512_mask_compressstoreu_ps((jaccard + nnz), non_zero_coefficients, tmp_v);

                nnz += _popcnt32_redef(_cvtmask16_u32(non_zero_coefficients));
                ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
            }

            j += 16;

            for (j = column_begin + ((diagonal - column_begin) / 16); j < diagonal; j++) {
                const auto j_neighbor_size = degrees[j];
                const auto j_neigbhors = cols + rows_vertex[j];
                if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                    !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                    auto intersection_value = preview::backend::intersection<Cpu>(i_neigbhors,
                                                                                  j_neigbhors,
                                                                                  i_neighbor_size,
                                                                                  j_neighbor_size);
                    if (intersection_value) {
                        jaccard[nnz] = static_cast<float>(intersection_value);
                        first_vertices[nnz] = i;
                        second_vertices[nnz] = j;
                        nnz++;
                        ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
                    }
                }
            }
        }
        else {
#endif
            for (j = column_begin; j < diagonal; j++) {
                const auto j_neighbor_size = degrees[j];
                const auto j_neigbhors = cols + rows_vertex[j];
                if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                    !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                    auto intersection_value = preview::backend::intersection<Cpu>(i_neigbhors,
                                                                                  j_neigbhors,
                                                                                  i_neighbor_size,
                                                                                  j_neighbor_size);
                    if (intersection_value) {
                        jaccard[nnz] = static_cast<float>(intersection_value);
                        first_vertices[nnz] = i;
                        second_vertices[nnz] = j;
                        nnz++;
                        ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
                    }
                }
            }
#if defined(__AVX512F__) && defined(DAAL_INTEL_CPP_COMPILER)
        }
#endif

        std::int32_t tmp_idx = column_begin;
        if (diagonal >= column_begin) {
            jaccard[nnz] = 1.0;
            first_vertices[nnz] = i;
            second_vertices[nnz] = diagonal;
            nnz++;
            ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
            tmp_idx = diagonal + 1;
        }
        j = tmp_idx;

#if defined(__AVX512F__) && defined(DAAL_INTEL_CPP_COMPILER)
        if (j < tmp_idx + ((column_end - tmp_idx) / 16) * 16) {
            //load_data(0)
            __m512i start_indices_j_v = _mm512_load_epi32(rows_vertex + j);
            __m512i end_indices_j_v_tmp = _mm512_load_epi32(rows_vertex + j + 1);
            __m512i end_indices_j_v = _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));

            __m512i n_j_start_v =
                _mm512_permutexvar_epi32(j_vertices_tmp1,
                                         _mm512_i32gather_epi32(start_indices_j_v, cols, 4));
            __m512i n_j_end_v =
                _mm512_permutexvar_epi32(j_vertices_tmp1,
                                         _mm512_i32gather_epi32(end_indices_j_v, cols, 4));

            for (; j + 16 < tmp_idx + ((column_end - tmp_idx) / 16) * 16;) {
                __m512i start_indices_j_v = _mm512_load_epi32(rows_vertex + j + 16);
                __m512i end_indices_j_v_tmp = _mm512_load_epi32(rows_vertex + j + 17);
                __m512i end_indices_j_v =
                    _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));

                __m512i n_j_start_v1 =
                    _mm512_permutexvar_epi32(j_vertices_tmp1,
                                             _mm512_i32gather_epi32(start_indices_j_v, cols, 4));
                __m512i n_j_end_v1 =
                    _mm512_permutexvar_epi32(j_vertices_tmp1,
                                             _mm512_i32gather_epi32(end_indices_j_v, cols, 4));

                __mmask16 cmpgt1 = _mm512_cmpgt_epi32_mask(n_i_start_v, n_j_end_v);
                __mmask16 cmpgt2 = _mm512_cmpgt_epi32_mask(n_j_start_v, n_i_end_v);

                __mmask16 worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
                ones_num = _popcnt32_redef(_cvtmask16_u32(worth_intersection));

                if (ones_num != 0) {
                    __m512i j_vertices_tmp2 = _mm512_set1_epi32(j);
                    __m512i j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                    _mm512_mask_compressstoreu_epi32((stack16_j_vertex),
                                                     worth_intersection,
                                                     j_vertices);

                    GRAPH_STACK_ALING(64) std::int32_t stack16_intersections[16] = { 0 };
                    for (std::int32_t s = 0; s < ones_num; s++) {
                        const auto j_neighbor_size = degrees[stack16_j_vertex[s]];
                        const auto j_neigbhors = cols + rows_vertex[stack16_j_vertex[s]];
                        stack16_intersections[s] =
                            preview::backend::intersection<Cpu>(i_neigbhors,
                                                                j_neigbhors,
                                                                i_neighbor_size,
                                                                j_neighbor_size);
                    }
                    __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                    j_vertices = _mm512_load_epi32(stack16_j_vertex);

                    __mmask16 non_zero_coefficients =
                        _mm512_test_epi32_mask(intersections_v, intersections_v);
                    _mm512_mask_compressstoreu_epi32((first_vertices + nnz),
                                                     non_zero_coefficients,
                                                     i_vertex);
                    _mm512_mask_compressstoreu_epi32((second_vertices + nnz),
                                                     non_zero_coefficients,
                                                     j_vertices);
                    __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                    _mm512_mask_compressstoreu_ps((jaccard + nnz), non_zero_coefficients, tmp_v);

                    nnz += _popcnt32_redef(_cvtmask16_u32(non_zero_coefficients));
                    ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
                }

                j += 16;

                n_j_start_v = n_j_start_v1;
                n_j_end_v = n_j_end_v1;
            }

            //process n data

            __mmask16 cmpgt1 = _mm512_cmpgt_epi32_mask(n_i_start_v, n_j_end_v);
            __mmask16 cmpgt2 = _mm512_cmpgt_epi32_mask(n_j_start_v, n_i_end_v);

            __mmask16 worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
            ones_num = _popcnt32_redef(_cvtmask16_u32(worth_intersection));

            if (ones_num != 0) {
                __m512i j_vertices_tmp2 = _mm512_set1_epi32(j);
                __m512i j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                _mm512_mask_compressstoreu_epi32((stack16_j_vertex),
                                                 worth_intersection,
                                                 j_vertices);

                GRAPH_STACK_ALING(64) std::int32_t stack16_intersections[16] = { 0 };
                for (std::int32_t s = 0; s < ones_num; s++) {
                    const auto j_neighbor_size = degrees[stack16_j_vertex[s]];
                    const auto j_neigbhors = cols + rows_vertex[stack16_j_vertex[s]];
                    stack16_intersections[s] = preview::backend::intersection<Cpu>(i_neigbhors,
                                                                                   j_neigbhors,
                                                                                   i_neighbor_size,
                                                                                   j_neighbor_size);
                }
                __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                j_vertices = _mm512_load_epi32(stack16_j_vertex);
                __mmask16 non_zero_coefficients =
                    _mm512_test_epi32_mask(intersections_v, intersections_v);
                _mm512_mask_compressstoreu_epi32((first_vertices + nnz),
                                                 non_zero_coefficients,
                                                 i_vertex);
                _mm512_mask_compressstoreu_epi32((second_vertices + nnz),
                                                 non_zero_coefficients,
                                                 j_vertices);
                __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                _mm512_mask_compressstoreu_ps((jaccard + nnz), non_zero_coefficients, tmp_v);

                nnz += _popcnt32_redef(_cvtmask16_u32(non_zero_coefficients));
                ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
            }

            j += 16;

            for (j = tmp_idx + ((column_end - tmp_idx) / 16) * 16; j < column_end; j++) {
                const auto j_neighbor_size = degrees[j];
                const auto j_neigbhors = cols + rows_vertex[j];
                if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                    !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                    auto intersection_value = preview::backend::intersection<Cpu>(i_neigbhors,
                                                                                  j_neigbhors,
                                                                                  i_neighbor_size,
                                                                                  j_neighbor_size);
                    if (intersection_value) {
                        jaccard[nnz] = static_cast<float>(intersection_value);
                        first_vertices[nnz] = i;
                        second_vertices[nnz] = j;
                        nnz++;
                        ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
                    }
                }
            }
        }
        else {
#endif
            for (j = tmp_idx; j < column_end; j++) {
                const auto j_neighbor_size = degrees[j];
                const auto j_neigbhors = cols + rows_vertex[j];
                if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                    !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                    auto intersection_value = preview::backend::intersection<Cpu>(i_neigbhors,
                                                                                  j_neigbhors,
                                                                                  i_neighbor_size,
                                                                                  j_neighbor_size);
                    if (intersection_value) {
                        jaccard[nnz] = static_cast<float>(intersection_value);
                        first_vertices[nnz] = i;
                        second_vertices[nnz] = j;
                        nnz++;
                        ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
                    }
                }
            }
#if defined(__AVX512F__) && defined(DAAL_INTEL_CPP_COMPILER)
        }
#endif
    }

    PRAGMA_OMP_SIMD
    for (int i = 0; i < nnz; i++) {
        const float notEqual(first_vertices[i] != second_vertices[i]);
        jaccard[i] = notEqual * (jaccard[i] / static_cast<float>(degrees[first_vertices[i]] +
                                                        degrees[second_vertices[i]] - jaccard[i])) +
                                                        (1.0f - notEqual) * jaccard[i];
    }

    vertex_similarity_result res(
        homogen_table::wrap(first_vertices, number_elements_in_block, 2, data_layout::column_major),
        homogen_table::wrap(jaccard, number_elements_in_block, 1, data_layout::column_major),
        nnz);
    return res;
}

} // namespace oneapi::dal::preview::jaccard::backend
