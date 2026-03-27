/* file: cpu_type.h */
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

#ifndef __CPU_TYPE_H__
#define __CPU_TYPE_H__

namespace daal
{
namespace internal
{
/**
 * Supported types of processor architectures
 */
enum CpuType
{
#if defined(TARGET_X86_64)
    sse2         = 0, /*!< Intel(R) Streaming SIMD Extensions 2 (Intel(R) SSE2) */
    firstCpuType = sse2,
    sse42        = 2, /*!< Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) */
    avx2         = 4, /*!< Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2) */
    avx512       = 6, /*!< Intel(R) Xeon(R) processors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) */
    lastCpuType  = avx512
#elif defined(TARGET_ARM)
    sve          = 0, /*!< ARM(R) processors based on Arm's Scalable Vector Extension (SVE) */
    firstCpuType = sve,
    lastCpuType  = sve
#elif defined(TARGET_RISCV64)
    rv64         = 0,
    firstCpuType = rv64,
    lastCpuType  = rv64
#endif
};

/**
 * Supported CPU features.
 * The features are defined as bit masks in order to allow for easy combination of features.
 * For example, (avx512_bf16 | avx512_vnni) will return a bit mask that indicates both the avx512_bf16
 * and avx512_vnni features are supported.
 * This allows for easy checking if a specific feature is supported by using a bitwise AND operation.
 * For example, (cpuFeatures & avx512_bf16) will return true if the avx512_bf16 feature is supported.
 */
enum CpuFeature
{
    unknown = 0ULL, /*!< Unknown features */
#if defined(TARGET_X86_64)
    sstep       = (1ULL << 0), /*!< Intel(R) SpeedStep */
    tb          = (1ULL << 1), /*!< Intel(R) Turbo Boost */
    avx512_bf16 = (1ULL << 2), /*!< AVX-512 bfloat16 */
    avx512_vnni = (1ULL << 3), /*!< AVX-512 Vector Neural Network Instructions (VNNI) */
    tb3         = (1ULL << 4), /*!< Intel(R) Turbo Boost Max 3.0 */
    amx_bf16    = (1ULL << 5), /*!< Intel(R) Advanced Matrix Extensions bfloat16 (AMX-BF16) */
#endif
};

/**
 * Precision hint for internal float32 matrix multiplications.
 *
 * This is a *hint*, not a hard override. The library decides which operations
 * are eligible for reduced precision; users express willingness to trade
 * accuracy for performance. Inspired by torch.set_float32_matmul_precision
 * and jax_default_matmul_precision.
 *
 * Levels
 * ------
 * highest (default)
 *   Always compute in the input dtype (float32 → sgemm, float64 → dgemm).
 *   Full IEEE-754, bit-exact and reproducible across hardware generations.
 *   Use when correctness or certification requirements preclude any rounding.
 *
 * high
 *   Allow oneDAL to use reduced-precision hardware kernels in operations
 *   where the library has determined the accuracy impact is bounded and
 *   acceptable for the algorithm. Currently enabled for:
 *     - float32 Euclidean distance GEMM (KNN brute-force, DBSCAN) on
 *       hardware with AMX-BF16, when all GEMM dimensions >= 64.
 *   Output dtype remains float32; only internal accumulation uses BF16.
 *   Accuracy impact is algorithm-specific: for Euclidean distance the
 *   empirical delta vs float32 baseline is < 1.2% on classification labels.
 *   Hardware without AMX-BF16 silently falls back to sgemm; no error raised.
 *
 * Future levels (not yet implemented)
 *   medium -- two-pass BF16 GEMM (2x BF16 → higher accuracy than single-pass,
 *             lower throughput than 'high'; analogous to PyTorch 'medium').
 *
 * Usage
 * -----
 *   Environment variable (set before loading oneDAL):
 *     ONEDAL_FLOAT32_MATMUL_PRECISION=HIGH
 *
 *   Programmatic (can be called at any time; takes effect on the next call):
 *     daal_set_float32_matmul_precision(daal::internal::Float32MatmulPrecision::high);
 *
 *   Query current setting:
 *     daal::internal::Float32MatmulPrecision p = daal_get_float32_matmul_precision();
 *
 *   Query hardware capability (independent of precision setting):
 *     bool amx = daal_has_amx_bf16();
 */
enum Float32MatmulPrecision
{
    highest = 0, /*!< Full IEEE-754 float32 arithmetic (default). No accuracy loss. */
    high    = 1, /*!< Allow BF16 kernels where oneDAL deems accuracy impact acceptable. */
};

} // namespace internal
} // namespace daal
#endif
