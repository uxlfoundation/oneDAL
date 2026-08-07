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

#if defined(TARGET_X86_64)
#include <immintrin.h>
#endif

#include <daal/src/services/service_defines.h>

namespace oneapi::dal::preview::backend {

template <typename Cpu>
ONEDAL_FORCEINLINE std::int64_t intersection(const std::int32_t *neigh_u,
                                             const std::int32_t *neigh_v,
                                             std::int32_t n_u,
                                             std::int32_t n_v) {
    std::int64_t total = 0;
    std::int32_t i_u = 0, i_v = 0;
    while (i_u < n_u && i_v < n_v) {
        if ((neigh_u[i_u] > neigh_v[n_v - 1]) || (neigh_v[i_v] > neigh_u[n_u - 1])) {
            return total;
        }
        if (neigh_u[i_u] == neigh_v[i_v])
            total++, i_u++, i_v++;
        else if (neigh_u[i_u] < neigh_v[i_v])
            i_u++;
        else if (neigh_u[i_u] > neigh_v[i_v])
            i_v++;
    }
    return total;
}

#if defined(DAAL_INTEL_CPP_COMPILER)
ONEDAL_FORCEINLINE std::int32_t _popcnt32_redef(const std::int32_t &x) {
    return _popcnt32(x);
}
#define GRAPH_STACK_ALING(x) __declspec(align(x))
#else
ONEDAL_FORCEINLINE std::int32_t _popcnt32_redef(const std::int32_t &x) {
    std::int32_t count = 0;
    std::int32_t a = x;
    while (a != 0) {
        a = a & (a - 1);
        count++;
    }
    return count;
}
#define GRAPH_STACK_ALING(x) \
    {                        \
    }
#endif

#if defined(TARGET_X86_64)
template <>
ONEDAL_FORCEINLINE std::int64_t intersection<dal::backend::cpu_dispatch_avx512>(
    const std::int32_t *neigh_u,
    const std::int32_t *neigh_v,
    std::int32_t n_u,
    std::int32_t n_v) {
    std::int64_t total = 0;
    std::int32_t i_u = 0, i_v = 0;
#if defined(__AVX512F__) && defined(DAAL_INTEL_CPP_COMPILER)
    while (i_u < (n_u / 16) * 16 && i_v < (n_v / 16) * 16) { // not in last n%16 elements
        // assumes neighbor list is ordered
        std::int32_t min_neigh_u = neigh_u[i_u];
        std::int32_t max_neigh_v = neigh_v[i_v + 15];

        if (min_neigh_u > max_neigh_v) {
            if (min_neigh_u > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 16;
            continue;
        }

        std::int32_t min_neigh_v = neigh_v[i_v];
        std::int32_t max_neigh_u = neigh_u[i_u + 15];
        if (min_neigh_v > max_neigh_u) {
            if (min_neigh_v > neigh_u[n_u - 1]) {
                return total;
            }
            i_u += 16;
            continue;
        }
        __m512i v_u = _mm512_loadu_si512((void *)(neigh_u + i_u)); // load 16 neighbors of u
        __m512i v_v = _mm512_loadu_si512((void *)(neigh_v + i_v)); // load 16 neighbors of v
        if (max_neigh_u >= max_neigh_v)
            i_v += 16;
        if (max_neigh_u <= max_neigh_v)
            i_u += 16;

        __mmask16 match = _mm512_cmpeq_epi32_mask(v_u, v_v);
        if (_mm512_mask2int(match) != 0xffff) { // shortcut case where all neighbors match
            __m512i circ1 = _mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
            __m512i circ2 = _mm512_set_epi32(1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2);
            __m512i circ3 = _mm512_set_epi32(2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3);
            __m512i circ4 = _mm512_set_epi32(3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4);
            __m512i circ5 = _mm512_set_epi32(4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5);
            __m512i circ6 = _mm512_set_epi32(5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6);
            __m512i circ7 = _mm512_set_epi32(6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7);
            __m512i circ8 = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
            __m512i circ9 = _mm512_set_epi32(8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9);
            __m512i circ10 = _mm512_set_epi32(9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10);
            __m512i circ11 = _mm512_set_epi32(10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11);
            __m512i circ12 = _mm512_set_epi32(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12);
            __m512i circ13 = _mm512_set_epi32(12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13);
            __m512i circ14 = _mm512_set_epi32(13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14);
            __m512i circ15 = _mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15);
            __m512i v_v1 = _mm512_permutexvar_epi32(circ1, v_v);
            __m512i v_v2 = _mm512_permutexvar_epi32(circ2, v_v);
            __m512i v_v3 = _mm512_permutexvar_epi32(circ3, v_v);
            __m512i v_v4 = _mm512_permutexvar_epi32(circ4, v_v);
            __m512i v_v5 = _mm512_permutexvar_epi32(circ5, v_v);
            __m512i v_v6 = _mm512_permutexvar_epi32(circ6, v_v);
            __m512i v_v7 = _mm512_permutexvar_epi32(circ7, v_v);
            __m512i v_v8 = _mm512_permutexvar_epi32(circ8, v_v);
            __m512i v_v9 = _mm512_permutexvar_epi32(circ9, v_v);
            __m512i v_v10 = _mm512_permutexvar_epi32(circ10, v_v);
            __m512i v_v11 = _mm512_permutexvar_epi32(circ11, v_v);
            __m512i v_v12 = _mm512_permutexvar_epi32(circ12, v_v);
            __m512i v_v13 = _mm512_permutexvar_epi32(circ13, v_v);
            __m512i v_v14 = _mm512_permutexvar_epi32(circ14, v_v);
            __m512i v_v15 = _mm512_permutexvar_epi32(circ15, v_v);
            __mmask16 tmp_match1 = _mm512_cmpeq_epi32_mask(v_u, v_v1); // find matches
            __mmask16 tmp_match2 = _mm512_cmpeq_epi32_mask(v_u, v_v2);
            __mmask16 tmp_match3 = _mm512_cmpeq_epi32_mask(v_u, v_v3);
            __mmask16 tmp_match4 = _mm512_cmpeq_epi32_mask(v_u, v_v4);
            __mmask16 tmp_match5 = _mm512_cmpeq_epi32_mask(v_u, v_v5);
            __mmask16 tmp_match6 = _mm512_cmpeq_epi32_mask(v_u, v_v6);
            __mmask16 tmp_match7 = _mm512_cmpeq_epi32_mask(v_u, v_v7);
            __mmask16 tmp_match8 = _mm512_cmpeq_epi32_mask(v_u, v_v8);
            __mmask16 tmp_match9 = _mm512_cmpeq_epi32_mask(v_u, v_v9);
            __mmask16 tmp_match10 = _mm512_cmpeq_epi32_mask(v_u, v_v10);
            __mmask16 tmp_match11 = _mm512_cmpeq_epi32_mask(v_u, v_v11);
            __mmask16 tmp_match12 = _mm512_cmpeq_epi32_mask(v_u, v_v12);
            __mmask16 tmp_match13 = _mm512_cmpeq_epi32_mask(v_u, v_v13);
            __mmask16 tmp_match14 = _mm512_cmpeq_epi32_mask(v_u, v_v14);
            __mmask16 tmp_match15 = _mm512_cmpeq_epi32_mask(v_u, v_v15);
            match = _mm512_kor(
                _mm512_kor(
                    _mm512_kor(_mm512_kor(match, tmp_match1), _mm512_kor(tmp_match2, tmp_match3)),
                    _mm512_kor(_mm512_kor(tmp_match4, tmp_match5),
                               _mm512_kor(tmp_match6, tmp_match7))),
                _mm512_kor(
                    _mm512_kor(_mm512_kor(tmp_match8, tmp_match9),
                               _mm512_kor(tmp_match10, tmp_match11)),
                    _mm512_kor(_mm512_kor(tmp_match12, tmp_match13),
                               _mm512_kor(tmp_match14, tmp_match15)))); // combine all matches
        }
        total += _popcnt32_redef(_mm512_mask2int(match)); //count number of matches
    }

    while (i_u < (n_u / 16) * 16 && i_v < n_v) {
        __m512i v_u = _mm512_loadu_si512((void *)(neigh_u + i_u));
        while (neigh_v[i_v] <= neigh_u[i_u + 15] && i_v < n_v) {
            __m512i tmp_v_v = _mm512_set1_epi32(neigh_v[i_v]);
            __mmask16 match = _mm512_cmpeq_epi32_mask(v_u, tmp_v_v);
            if (_mm512_mask2int(match))
                total++;
            i_v++;
        }
        i_u += 16;
    }
    while (i_v < (n_v / 16) * 16 && i_u < n_u) {
        __m512i v_v = _mm512_loadu_si512((void *)(neigh_v + i_v));
        while (neigh_u[i_u] <= neigh_v[i_v + 15] && i_u < n_u) {
            __m512i tmp_v_u = _mm512_set1_epi32(neigh_u[i_u]);
            __mmask16 match = _mm512_cmpeq_epi32_mask(v_v, tmp_v_u);
            if (_mm512_mask2int(match))
                total++;
            i_u++;
        }
        i_v += 16;
    }

    while (i_u <= (n_u - 8) && i_v <= (n_v - 8)) { // not in last n%8 elements
        // assumes neighbor list is ordered
        std::int32_t min_neigh_u = neigh_u[i_u];
        std::int32_t max_neigh_v = neigh_v[i_v + 7];

        if (min_neigh_u > max_neigh_v) {
            if (min_neigh_u > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 8;
            continue;
        }
        std::int32_t max_neigh_u = neigh_u[i_u + 7];
        std::int32_t min_neigh_v = neigh_v[i_v];
        if (min_neigh_v > max_neigh_u) {
            if (min_neigh_v > neigh_u[n_u - 1]) {
                return total;
            }
            i_u += 8;
            continue;
        }
        __m256i v_u = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(neigh_u + i_u)); // load 8 neighbors of u
        __m256i v_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(neigh_v + i_v)); // load 8 neighbors of v

        if (max_neigh_u >= max_neigh_v)
            i_v += 8;
        if (max_neigh_u <= max_neigh_v)
            i_u += 8;

        __mmask8 match = _mm256_cmpeq_epi32_mask(v_u, v_v);
        if (_cvtmask8_u32(match) != 0xff) { // shortcut case where all neighbors match
            __m256i circ1 = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
            __m256i circ2 = _mm256_set_epi32(1, 0, 7, 6, 5, 4, 3, 2);
            __m256i circ3 = _mm256_set_epi32(2, 1, 0, 7, 6, 5, 4, 3);
            __m256i circ4 = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
            __m256i circ5 = _mm256_set_epi32(4, 3, 2, 1, 0, 7, 6, 5);
            __m256i circ6 = _mm256_set_epi32(5, 4, 3, 2, 1, 0, 7, 6);
            __m256i circ7 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);

            __m256i v_v1 = _mm256_permutexvar_epi32(circ1, v_v);
            __m256i v_v2 = _mm256_permutexvar_epi32(circ2, v_v);
            __m256i v_v3 = _mm256_permutexvar_epi32(circ3, v_v);
            __m256i v_v4 = _mm256_permutexvar_epi32(circ4, v_v);
            __m256i v_v5 = _mm256_permutexvar_epi32(circ5, v_v);
            __m256i v_v6 = _mm256_permutexvar_epi32(circ6, v_v);
            __m256i v_v7 = _mm256_permutexvar_epi32(circ7, v_v);

            __mmask8 tmp_match1 = _mm256_cmpeq_epi32_mask(v_u, v_v1); // find matches
            __mmask8 tmp_match2 = _mm256_cmpeq_epi32_mask(v_u, v_v2);
            __mmask8 tmp_match3 = _mm256_cmpeq_epi32_mask(v_u, v_v3);
            __mmask8 tmp_match4 = _mm256_cmpeq_epi32_mask(v_u, v_v4);
            __mmask8 tmp_match5 = _mm256_cmpeq_epi32_mask(v_u, v_v5);
            __mmask8 tmp_match6 = _mm256_cmpeq_epi32_mask(v_u, v_v6);
            __mmask8 tmp_match7 = _mm256_cmpeq_epi32_mask(v_u, v_v7);

            match = _kor_mask8(
                _kor_mask8(_kor_mask8(match, tmp_match1), _kor_mask8(tmp_match2, tmp_match3)),
                _kor_mask8(_kor_mask8(tmp_match4, tmp_match5),
                           _kor_mask8(tmp_match6, tmp_match7))); // combine all matches
        }
        total += _popcnt32_redef(_cvtmask8_u32(match)); //count number of matches
    }
    if (i_u <= (n_u - 8) && i_v < n_v) {
        __m256i v_u = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(neigh_u + i_u));
        while (neigh_v[i_v] <= neigh_u[i_u + 7] && i_v < n_v) {
            __m256i tmp_v_v = _mm256_set1_epi32(neigh_v[i_v]);
            __mmask8 match = _mm256_cmpeq_epi32_mask(v_u, tmp_v_v);
            if (_cvtmask8_u32(match))
                total++;
            i_v++;
        }
        i_u += 8;
    }
    if (i_v <= (n_v - 8) && i_u < n_u) {
        __m256i v_v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(neigh_v + i_v));
        while (neigh_u[i_u] <= neigh_v[i_v + 7] && i_u < n_u) {
            __m256i tmp_v_u = _mm256_set1_epi32(neigh_u[i_u]);
            __mmask8 match = _mm256_cmpeq_epi32_mask(v_v, tmp_v_u);
            if (_cvtmask8_u32(match))
                total++;
            i_u++;
        }
        i_v += 8;
    }

    while (i_u <= (n_u - 4) && i_v <= (n_v - 4)) { // not in last n%8 elements
        // assumes neighbor list is ordered
        std::int32_t min_neigh_u = neigh_u[i_u];
        std::int32_t max_neigh_v = neigh_v[i_v + 3];

        if (min_neigh_u > max_neigh_v) {
            if (min_neigh_u > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 4;
            continue;
        }
        std::int32_t min_neigh_v = neigh_v[i_v];
        std::int32_t max_neigh_u = neigh_u[i_u + 3];
        if (min_neigh_v > max_neigh_u) {
            if (min_neigh_v > neigh_u[n_u - 1]) {
                return total;
            }
            i_u += 4;
            continue;
        }
        __m128i v_u = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_u + i_u)); // load 8 neighbors of u
        __m128i v_v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_v + i_v)); // load 8 neighbors of v

        if (max_neigh_u >= max_neigh_v)
            i_v += 4;
        if (max_neigh_u <= max_neigh_v)
            i_u += 4;

        __mmask8 match = _mm_cmpeq_epi32_mask(v_u, v_v);
        if (_cvtmask8_u32(match) != 0xf) { // shortcut case where all neighbors match
            __m128i v_v1 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(0, 3, 2, 1));
            __m128i v_v2 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(1, 0, 3, 2));
            __m128i v_v3 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(2, 1, 0, 3));

            __mmask8 tmp_match1 = _mm_cmpeq_epi32_mask(v_u, v_v1); // find matches
            __mmask8 tmp_match2 = _mm_cmpeq_epi32_mask(v_u, v_v2);
            __mmask8 tmp_match3 = _mm_cmpeq_epi32_mask(v_u, v_v3);

            match = _kor_mask8(_kor_mask8(match, tmp_match1),
                               _kor_mask8(tmp_match2, tmp_match3)); // combine all matches
        }
        total += _popcnt32_redef(_cvtmask8_u32(match)); //count number of matches
    }
    if (i_u <= (n_u - 4) && i_v < n_v) {
        __m128i v_u = _mm_loadu_si128(reinterpret_cast<const __m128i *>(neigh_u + i_u));
        while (neigh_v[i_v] <= neigh_u[i_u + 3] && i_v < n_v) {
            __m128i tmp_v_v = _mm_set1_epi32(neigh_v[i_v]);
            __mmask8 match = _mm_cmpeq_epi32_mask(v_u, tmp_v_v);
            if (_cvtmask8_u32(match))
                total++;
            i_v++;
        }
        i_u += 4;
    }
    if (i_v <= (n_v - 4) && i_u < n_u) {
        __m128i v_v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(neigh_v + i_v));
        while (neigh_u[i_u] <= neigh_v[i_v + 3] && i_u < n_u) {
            __m128i tmp_v_u = _mm_set1_epi32(neigh_u[i_u]);
            __mmask8 match = _mm_cmpeq_epi32_mask(v_v, tmp_v_u);
            if (_cvtmask8_u32(match))
                total++;
            i_u++;
        }
        i_v += 4;
    }
#endif
    while (i_u < n_u && i_v < n_v) {
        if ((neigh_u[i_u] > neigh_v[n_v - 1]) || (neigh_v[i_v] > neigh_u[n_u - 1])) {
            return total;
        }
        if (neigh_u[i_u] == neigh_v[i_v])
            total++, i_u++, i_v++;
        else if (neigh_u[i_u] < neigh_v[i_v])
            i_u++;
        else if (neigh_u[i_u] > neigh_v[i_v])
            i_v++;
    }
    return total;
}

template <>
ONEDAL_FORCEINLINE std::int64_t intersection<dal::backend::cpu_dispatch_avx2>(
    const std::int32_t *neigh_u,
    const std::int32_t *neigh_v,
    std::int32_t n_u,
    std::int32_t n_v) {
    std::int64_t total = 0;
    std::int32_t i_u = 0, i_v = 0;
#if defined(__AVX2__) && defined(DAAL_INTEL_CPP_COMPILER)
    const std::int32_t n_u_8_end = n_u - 8;
    const std::int32_t n_v_8_end = n_v - 8;
    while (i_u <= n_u_8_end && i_v <= n_v_8_end) {
        const std::int32_t min_neigh_u = neigh_u[i_u];
        const std::int32_t max_neigh_v = neigh_v[i_v + 7];

        if (min_neigh_u > max_neigh_v) {
            if (min_neigh_u > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 8;
            continue;
        }

        const std::int32_t max_neigh_u = neigh_u[i_u + 7]; // assumes neighbor list is ordered
        const std::int32_t min_neigh_v = neigh_v[i_v];

        if (min_neigh_v > max_neigh_u) {
            if (min_neigh_v > neigh_u[n_u - 1]) {
                return total;
            }
            i_u += 8;
            continue;
        }

        __m256i v_u = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(neigh_u + i_u)); // load 8 neighbors of u
        __m256i v_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(neigh_v + i_v)); // load 8 neighbors of v

        i_v = (max_neigh_u >= max_neigh_v) ? i_v + 8 : i_v;
        i_u = (max_neigh_u <= max_neigh_v) ? i_u + 8 : i_u;

        __m256i match = _mm256_cmpeq_epi32(v_u, v_v);
        unsigned int scalar_match = _mm256_movemask_ps(_mm256_castsi256_ps(match));

        if (scalar_match != 255) { // shortcut case where all neighbors match
            __m256i circ1 = _mm256_set_epi32(0,
                                             7,
                                             6,
                                             5,
                                             4,
                                             3,
                                             2,
                                             1); // all possible circular shifts for 16 elements
            __m256i circ2 = _mm256_set_epi32(1, 0, 7, 6, 5, 4, 3, 2);
            __m256i circ3 = _mm256_set_epi32(2, 1, 0, 7, 6, 5, 4, 3);
            __m256i circ4 = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
            __m256i circ5 = _mm256_set_epi32(4, 3, 2, 1, 0, 7, 6, 5);
            __m256i circ6 = _mm256_set_epi32(5, 4, 3, 2, 1, 0, 7, 6);
            __m256i circ7 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);

            __m256i v_v1 = _mm256_permutevar8x32_epi32(v_v, circ1);
            __m256i v_v2 = _mm256_permutevar8x32_epi32(v_v, circ2);
            __m256i v_v3 = _mm256_permutevar8x32_epi32(v_v, circ3);
            __m256i v_v4 = _mm256_permutevar8x32_epi32(v_v, circ4);
            __m256i v_v5 = _mm256_permutevar8x32_epi32(v_v, circ5);
            __m256i v_v6 = _mm256_permutevar8x32_epi32(v_v, circ6);
            __m256i v_v7 = _mm256_permutevar8x32_epi32(v_v, circ7);

            __m256i tmp_match1 = _mm256_cmpeq_epi32(v_u, v_v1); // find matches
            __m256i tmp_match2 = _mm256_cmpeq_epi32(v_u, v_v2);
            __m256i tmp_match3 = _mm256_cmpeq_epi32(v_u, v_v3);
            __m256i tmp_match4 = _mm256_cmpeq_epi32(v_u, v_v4);
            __m256i tmp_match5 = _mm256_cmpeq_epi32(v_u, v_v5);
            __m256i tmp_match6 = _mm256_cmpeq_epi32(v_u, v_v6);
            __m256i tmp_match7 = _mm256_cmpeq_epi32(v_u, v_v7);

            unsigned int scalar_match1 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match1));
            unsigned int scalar_match2 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match2));
            unsigned int scalar_match3 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match3));
            unsigned int scalar_match4 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match4));
            unsigned int scalar_match5 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match5));
            unsigned int scalar_match6 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match6));
            unsigned int scalar_match7 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match7));
            unsigned int final_match = scalar_match | scalar_match1 | scalar_match2 |
                                       scalar_match3 | scalar_match4 | scalar_match5 |
                                       scalar_match6 | scalar_match7;

            total += _popcnt32_redef(final_match);
        }
        else {
            total += 8; //count number of matches
        }
    }

    for (; i_u <= n_u_8_end && i_v < n_v; i_u += 8) {
        __m256i v_u = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(neigh_u + i_u));

        const std::int32_t neighu_iu = neigh_u[i_u + 7];
        for (; neigh_v[i_v] <= neighu_iu && i_v < n_v; i_v++) {
            __m256i tmp_v_v = _mm256_set1_epi32(neigh_v[i_v]);

            __m256i match = _mm256_cmpeq_epi32(v_u, tmp_v_v);
            unsigned int scalar_match = _mm256_movemask_ps(_mm256_castsi256_ps(match));
            total = scalar_match != 0 ? total + 1 : total;
        }
    }
    for (; i_v <= n_v_8_end && i_u < n_u; i_v += 8) {
        __m256i v_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(neigh_v + i_v)); // load 8 neighbors of v
        const std::int32_t neighv_iv = neigh_v[i_v + 7];
        for (; neigh_u[i_u] <= neighv_iv && i_u < n_u; i_u++) {
            __m256i tmp_v_u = _mm256_set1_epi32(neigh_u[i_u]);
            __m256i match = _mm256_cmpeq_epi32(v_v, tmp_v_u);
            unsigned int scalar_match = _mm256_movemask_ps(_mm256_castsi256_ps(match));
            total = scalar_match != 0 ? total + 1 : total;
        }
    }

    const std::int32_t n_u_4_end = n_u - 4;
    const std::int32_t n_v_4_end = n_v - 4;

    while (i_u <= n_u_4_end && i_v <= n_v_4_end) { // not in last n%8 elements
        // assumes neighbor list is ordered
        std::int32_t min_neigh_u = neigh_u[i_u];
        std::int32_t max_neigh_v = neigh_v[i_v + 3];

        if (min_neigh_u > max_neigh_v) {
            if (min_neigh_u > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 4;
            continue;
        }
        std::int32_t min_neigh_v = neigh_v[i_v];
        std::int32_t max_neigh_u = neigh_u[i_u + 3];
        if (min_neigh_v > max_neigh_u) {
            if (min_neigh_v > neigh_u[n_u - 1]) {
                return total;
            }
            i_u += 4;
            continue;
        }

        __m128i v_u = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_u + i_u)); // load 8 neighbors of u
        __m128i v_v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_v + i_v)); // load 8 neighbors of v

        i_v = (max_neigh_u >= max_neigh_v) ? i_v + 4 : i_v;
        i_u = (max_neigh_u <= max_neigh_v) ? i_u + 4 : i_u;

        __m128i match = _mm_cmpeq_epi32(v_u, v_v);
        unsigned int scalar_match = _mm_movemask_ps(_mm_castsi128_ps(match));

        if (scalar_match != 15) { // shortcut case where all neighbors match
            __m128i v_v1 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(0, 3, 2, 1));
            __m128i v_v2 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(1, 0, 3, 2));
            __m128i v_v3 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(2, 1, 0, 3));

            __m128i tmp_match1 = _mm_cmpeq_epi32(v_u, v_v1); // find matches
            __m128i tmp_match2 = _mm_cmpeq_epi32(v_u, v_v2);
            __m128i tmp_match3 = _mm_cmpeq_epi32(v_u, v_v3);

            unsigned int scalar_match1 = _mm_movemask_ps(_mm_castsi128_ps(tmp_match1));
            unsigned int scalar_match2 = _mm_movemask_ps(_mm_castsi128_ps(tmp_match2));
            unsigned int scalar_match3 = _mm_movemask_ps(_mm_castsi128_ps(tmp_match3));

            unsigned int final_match = scalar_match | scalar_match1 | scalar_match2 | scalar_match3;

            total += _popcnt32_redef(final_match);
        }
        else {
            total += 4; //count number of matches
        }
    }

    if (i_u <= n_u_4_end && i_v < n_v) {
        __m128i v_u = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_u + i_u)); // load 8 neighbors of u
        const std::int32_t neighu_iu = neigh_u[i_u + 3];
        for (; neigh_v[i_v] <= neighu_iu && i_v < n_v; i_v++) {
            __m128i tmp_v_v = _mm_set1_epi32(neigh_v[i_v]);
            __m128i match = _mm_cmpeq_epi32(v_u, tmp_v_v);
            unsigned int scalar_match = _mm_movemask_ps(_mm_castsi128_ps(match));
            total = scalar_match != 0 ? total + 1 : total;
        }
        i_u += 4;
    }
    if (i_v <= n_v_4_end && i_u < n_u) {
        __m128i v_v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_v + i_v)); // load 8 neighbors of v
        const std::int32_t neighv_iv = neigh_v[i_v + 3];
        for (; neigh_u[i_u] <= neighv_iv && i_u < n_u; i_u++) {
            __m128i tmp_v_u = _mm_set1_epi32(neigh_u[i_u]);
            __m128i match = _mm_cmpeq_epi32(v_v, tmp_v_u);
            unsigned int scalar_match = _mm_movemask_ps(_mm_castsi128_ps(match));
            total = scalar_match != 0 ? total + 1 : total;
        }
        i_v += 4;
    }
#endif
    while (i_u < n_u && i_v < n_v) {
        if ((neigh_u[i_u] > neigh_v[n_v - 1]) || (neigh_v[i_v] > neigh_u[n_u - 1])) {
            return total;
        }
        if (neigh_u[i_u] == neigh_v[i_v])
            total++, i_u++, i_v++;
        else if (neigh_u[i_u] < neigh_v[i_v])
            i_u++;
        else if (neigh_u[i_u] > neigh_v[i_v])
            i_v++;
    }
    return total;
}
#endif

} // namespace oneapi::dal::preview::backend
