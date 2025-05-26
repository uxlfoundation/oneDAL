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

#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/backend/primitives/rng/utils.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_types.hpp"

#include <daal/include/algorithms/engines/mt2203/mt2203.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/include/algorithms/engines/mrg32k3a/mrg32k3a.h>
#include <daal/include/algorithms/engines/philox4x32x10/philox4x32x10.h>
#include <daal/include/algorithms/engines/mt19937/mt19937.h>

#include <oneapi/mkl.hpp>

namespace mkl = oneapi::mkl;
namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

/// Abstract base class for all random number generators (RNGs).
/// It defines a common interface for working with different types of RNGs, including methods
/// for retrieving the engine method and skipping ahead in the random number sequence.
class gen_base {
public:
    virtual ~gen_base() = default;

    /// Method to retrieve the engine method.
    /// @return The engine method as an enum value of `engine_type_internal`.
    virtual engine_type_internal get_engine_type() const = 0;

    /// Method to skip ahead in the random number sequence.
    /// @param[in] nSkip The number of steps to skip in the generator sequence.
    virtual void skip_ahead_gpu(std::int64_t nSkip) = 0;
};

/// Implementation of the mt2203 random number generator for GPU.
/// This class encapsulates the mt2203 engine and provides an interface to use it
/// on devices supporting SYCL.
class gen_mt2203 : public gen_base {
public:
    explicit gen_mt2203() = delete;

    /// Constructor that initializes the mt2203 generator for use on the GPU.
    /// @param[in] queue The SYCL queue to manage device operations.
    /// @param[in] seed The initial seed for the generator.
    gen_mt2203(sycl::queue queue, std::int64_t seed, std::int64_t engine_idx = 0)
            : _gen(queue, seed, engine_idx) {}

    /// Returns the engine method for mt2203.
    /// @return The `mt2203` engine method as an enum value of `engine_type_internal`.
    engine_type_internal get_engine_type() const override {
        return engine_type_internal::mt2203;
    }

    /// Skips ahead in the random number sequence for mt2203 on the GPU.
    /// Currently, the skip functionality is not implemented.
    /// @param[in] nSkip The number of steps to skip in the sequence.
    void skip_ahead_gpu(std::int64_t nSkip) override {
        //skip;
    }

    /// Retrieves a pointer to the underlying mt2203 generator.
    /// @return A pointer to the `mt2203` RNG.
    oneapi::mkl::rng::mt2203* get() {
        return &_gen;
    }

protected:
    oneapi::mkl::rng::mt2203 _gen;
};

/// Implementation of the philox4x32x10 random number generator for GPU.
/// This class encapsulates the philox4x32x10 engine and provides an interface to use it
/// on devices supporting SYCL.
class gen_philox : public gen_base {
public:
    explicit gen_philox() = delete;

    /// Constructor that initializes the philox4x32x10 generator for use on the GPU.
    /// @param[in] queue The SYCL queue to manage device operations.
    /// @param[in] seed The initial seed for the generator.
    gen_philox(sycl::queue queue, std::int64_t seed) : _gen(queue, seed) {}

    /// Returns the engine method for philox4x32x10.
    /// @return The `philox4x32x10` engine method as an enum value of `engine_type_internal`.
    engine_type_internal get_engine_type() const override {
        return engine_type_internal::philox4x32x10;
    }

    /// Skips ahead in the random number sequence for philox4x32x10 on the GPU.
    /// @param[in] nSkip The number of steps to skip in the sequence.
    void skip_ahead_gpu(std::int64_t nSkip) override {
        skip_ahead(_gen, nSkip);
    }

    /// Retrieves a pointer to the underlying philox4x32x10 generator.
    /// @return A pointer to the `philox4x32x10` RNG.
    oneapi::mkl::rng::philox4x32x10* get() {
        return &_gen;
    }

protected:
    oneapi::mkl::rng::philox4x32x10 _gen;
};

/// Implementation of the mrg32k3a random number generator for GPU.
/// This class encapsulates the mrg32k3a engine and provides an interface to use it
/// on devices supporting SYCL.
class gen_mrg32k : public gen_base {
public:
    explicit gen_mrg32k() = delete;

    /// Constructor that initializes the mrg32k3a generator for use on the GPU.
    /// @param[in] queue The SYCL queue to manage device operations.
    /// @param[in] seed The initial seed for the generator.
    gen_mrg32k(sycl::queue queue, std::int64_t seed) : _gen(queue, seed) {}

    /// Returns the engine method for mrg32k3a.
    /// @return The `mrg32k3a` engine method as an enum value of `engine_type_internal`.
    engine_type_internal get_engine_type() const override {
        return engine_type_internal::mrg32k3a;
    }

    /// Skips ahead in the random number sequence for mrg32k3a on the GPU.
    /// @param[in] nSkip The number of steps to skip in the sequence.
    void skip_ahead_gpu(std::int64_t nSkip) override {
        skip_ahead(_gen, nSkip);
    }

    /// Retrieves a pointer to the underlying mrg32k3a generator.
    /// @return A pointer to the `mrg32k3a` RNG.
    oneapi::mkl::rng::mrg32k3a* get() {
        return &_gen;
    }

protected:
    oneapi::mkl::rng::mrg32k3a _gen;
};

/// Implementation of the mt19937 random number generator for GPU.
/// This class encapsulates the mt19937 engine and provides an interface to use it
/// on devices supporting SYCL.
class gen_mt19937 : public gen_base {
public:
    explicit gen_mt19937() = delete;

    /// Constructor that initializes the mt19937 generator for use on the GPU.
    /// @param[in] queue The SYCL queue to manage device operations.
    /// @param[in] seed The initial seed for the generator.
    gen_mt19937(sycl::queue queue, std::int64_t seed) : _gen(queue, seed) {}

    /// Returns the engine method for mt19937.
    /// @return The `mt19937` engine method as an enum value of `engine_type_internal`.
    engine_type_internal get_engine_type() const override {
        return engine_type_internal::mt19937;
    }

    /// Skips ahead in the random number sequence for mt19937 on the GPU.
    /// @param[in] nSkip The number of steps to skip in the sequence.
    void skip_ahead_gpu(std::int64_t nSkip) override {
        skip_ahead(_gen, nSkip);
    }

    /// Retrieves a pointer to the underlying mt19937 generator.
    /// @return A pointer to the `mt19937` RNG.
    oneapi::mkl::rng::mt19937* get() {
        return &_gen;
    }

protected:
    oneapi::mkl::rng::mt19937 _gen;
};

/// Implementation of the mcg59 random number generator for GPU.
/// This class encapsulates the mcg59 engine and provides an interface to use it
/// on devices supporting SYCL.
class gen_mcg59 : public gen_base {
public:
    explicit gen_mcg59() = delete;

    /// Constructor that initializes the mcg59 generator for use on the GPU.
    /// @param[in] queue The SYCL queue to manage device operations.
    /// @param[in] seed The initial seed for the generator.
    gen_mcg59(sycl::queue queue, std::int64_t seed) : _gen(queue, seed) {}

    /// Returns the engine method for mcg59.
    /// @return The `mcg59` engine method as an enum value of `engine_type_internal`.
    engine_type_internal get_engine_type() const override {
        return engine_type_internal::mcg59;
    }

    /// Skips ahead in the random number sequence for mcg59 on the GPU.
    /// @param[in] nSkip The number of steps to skip in the sequence.
    void skip_ahead_gpu(std::int64_t nSkip) override {
        skip_ahead(_gen, nSkip);
    }

    /// Retrieves a pointer to the underlying mcg59 generator.
    /// @return A pointer to the `mcg59` RNG.
    oneapi::mkl::rng::mcg59* get() {
        return &_gen;
    }

protected:
    oneapi::mkl::rng::mcg59 _gen;
};

/// A class that provides a unified interface for random number generation on both CPU and GPU devices.
///
/// This class serves as a wrapper for random number generators (RNGs) that supports different engine types,
/// enabling efficient random number generation on heterogeneous platforms using SYCL. It integrates a host
/// (CPU) engine and a device (GPU) engine, allowing operations to be executed seamlessly on the appropriate
/// device.
///
/// The class provides functionality to skip ahead in the RNG sequence, retrieve engine states, and
/// manage host and device engines independently. Support for `skip_ahead` on GPU is currently limited for
/// some engine types.
class device_engine {
public:
    /// @param[in] queue   The SYCL queue used to manage device operations.
    /// @param[in] seed    The initial seed for the random number generator. Defaults to `777`.
    /// @param[in] method  The engine method. Defaults to `engine_type_internal::mt2203`.
    device_engine(sycl::queue& queue,
                  std::int64_t seed = 777,
                  engine_type_internal method = engine_type_internal::mt2203,
                  std::int64_t idx = 0)
            : q(queue) {
        switch (method) {
            case engine_type_internal::mt2203:
                host_engine_ = daal::algorithms::engines::mt2203::Batch<>::create(seed);
                dpc_engine_ = std::make_shared<gen_mt2203>(queue, seed, idx);
                break;
            case engine_type_internal::mcg59:
                host_engine_ = daal::algorithms::engines::mcg59::Batch<>::create(seed);
                dpc_engine_ = std::make_shared<gen_mcg59>(queue, seed);
                break;
            case engine_type_internal::mrg32k3a:
                host_engine_ = daal::algorithms::engines::mrg32k3a::Batch<>::create(seed);
                dpc_engine_ = std::make_shared<gen_mrg32k>(queue, seed);
                break;
            case engine_type_internal::philox4x32x10:
                host_engine_ = daal::algorithms::engines::philox4x32x10::Batch<>::create(seed);
                dpc_engine_ = std::make_shared<gen_philox>(queue, seed);
                break;
            case engine_type_internal::mt19937:
                host_engine_ = daal::algorithms::engines::mt19937::Batch<>::create(seed);
                dpc_engine_ = std::make_shared<gen_mt19937>(queue, seed);
                break;
            default: throw std::invalid_argument("Unsupported engine type 1");
        }
        impl_ =
            dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(host_engine_.get());
        if (!impl_) {
            throw std::domain_error("RNG engine is not supported");
        }
    }

    // Copy constructor
    device_engine(const device_engine&) = default;

    // Copy assignment operator
    device_engine& operator=(const device_engine&) = default;

    // Move constructor
    device_engine(device_engine&& other) noexcept = default;

    // Move assignment operator
    device_engine& operator=(device_engine&& other) noexcept = default;

    /// Destructor.
    ~device_engine() = default;

    /// Retrieves the state of the host rng engine(DAAL).
    /// @return Pointer to the host engine state.
    void* get_host_engine_state() const {
        return impl_->getState();
    }

    /// Retrieves the base pointer of the device rng engine.
    /// @return Shared pointer to the device rng engine base.
    auto get_device_engine_base_ptr() {
        return dpc_engine_;
    }

    /// Advances the rng sequence on the CPU by a specified number of steps.
    /// @param[in] nSkip The number of steps to skip.
    void skip_ahead_cpu(size_t nSkip) {
        host_engine_->skipAhead(nSkip);
    }

    /// Advances the rng sequence on the GPU by a specified number of steps.
    /// @param[in] nSkip The number of steps to skip.
    void skip_ahead_gpu(size_t nSkip) {
        dpc_engine_->skip_ahead_gpu(nSkip);
    }

    /// Advances the rng sequence on both CPU and GPU by a specified number of steps.
    /// @param[in] nSkip The number of steps to skip.
    void skip_ahead(size_t nSkip) {
        skip_ahead_cpu(nSkip);
        skip_ahead_gpu(nSkip);
    }

    /// Retrieves the SYCL queue associated with this rng engine.
    /// @return Reference to the SYCL queue.
    sycl::queue& get_queue() {
        return q;
    }

private:
    sycl::queue q;
    daal::algorithms::engines::EnginePtr host_engine_;
    std::shared_ptr<gen_base> dpc_engine_;
    daal::algorithms::engines::internal::BatchBaseImpl* impl_;
};

/// Generates uniformly distributed random numbers on the CPU.
/// @tparam Type The data type of the generated numbers.
/// @param[in] count The number of random numbers to generate.
/// @param[out] dst Pointer to the output buffer.
/// @param[in] engine_ Reference to the device engine.
/// @param[in] a The lower bound of the uniform distribution.
/// @param[in] b The upper bound of the uniform distribution.
template <typename Type>
void uniform(std::int64_t count, Type* dst, device_engine& engine_, Type a, Type b) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    auto state = engine_.get_host_engine_state();
    uniform_dispatcher::uniform_by_cpu<Type>(count, dst, state, a, b);
    engine_.skip_ahead_gpu(count);
}

/// Generates a random permutation of elements without replacement.
/// @tparam Type The data type of the elements.
/// @param[in] count The number of elements to generate.
/// @param[out] dst Pointer to the output buffer.
/// @param[out] buffer Temporary buffer used for computations.
/// @param[in] engine_ Reference to the device engine.
/// @param[in] a The lower bound of the range.
/// @param[in] b The upper bound of the range.
template <typename Type>
void uniform_without_replacement(std::int64_t count,
                                 Type* dst,
                                 device_engine& engine_,
                                 Type a,
                                 Type b) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    void* state = engine_.get_host_engine_state();
    uniform_dispatcher::uniform_without_replacement_by_cpu<Type>(count, dst, state, a, b);
    engine_.skip_ahead_gpu(count);
}

/// Shuffles an array using random swaps.
/// @tparam Type The data type of the array elements.
/// @param[in] count The number of elements to shuffle.
/// @param[in, out] dst Pointer to the array to be shuffled.
/// @param[in] engine_ Reference to the device engine.
template <typename Type, typename = std::enable_if_t<std::is_integral_v<Type>>>
void shuffle(std::int64_t count, Type* dst, device_engine& engine_) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    Type idx[2];
    void* state = engine_.get_host_engine_state();
    for (std::int64_t i = 0; i < count; ++i) {
        uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
        std::swap(dst[idx[0]], dst[idx[1]]);
    }
    engine_.skip_ahead_gpu(count);
}

/// Generates uniformly distributed random numbers on the GPU.
/// @tparam Type The data type of the generated numbers.
/// @param[in] queue The SYCL queue for device execution.
/// @param[in] count The number of random numbers to generate.
/// @param[out] dst Pointer to the output buffer.
/// @param[in] engine_ Reference to the device engine.
/// @param[in] a The lower bound of the uniform distribution.
/// @param[in] b The upper bound of the uniform distribution.
/// @param[in] deps Dependencies for the SYCL event.
template <typename Type>
sycl::event uniform(sycl::queue& queue,
                    std::int64_t count,
                    Type* dst,
                    device_engine& engine_,
                    Type a,
                    Type b,
                    const event_vector& deps = {});

/// Generates a random permutation of elements without replacement on the GPU.
/// @tparam Type The data type of the elements.
/// @param[in] queue The SYCL queue for device execution.
/// @param[in] count The number of elements to generate.
/// @param[out] dst Pointer to the output buffer.
/// @param[out] buffer Temporary buffer used for computations.
/// @param[in] engine_ Reference to the device engine.
/// @param[in] a The lower bound of the range.
/// @param[in] b The upper bound of the range.
/// @param[in] deps Dependencies for the SYCL event.
template <typename Type>
sycl::event uniform_without_replacement(sycl::queue& queue,
                                        std::int64_t count,
                                        Type* dst,
                                        device_engine& engine_,
                                        Type a,
                                        Type b,
                                        const event_vector& deps = {});

/// Shuffles an array using random swaps on the GPU.
/// @tparam Type The data type of the array elements.
/// @param[in] queue The SYCL queue for device execution.
/// @param[in] count The number of elements to shuffle.
/// @param[in, out] dst Pointer to the array to be shuffled.
/// @param[in] engine_ Reference to the device engine.
/// @param[in] deps Dependencies for the SYCL event.
template <typename Type>
sycl::event shuffle(sycl::queue& queue,
                    std::int64_t count,
                    Type* dst,
                    device_engine& engine_,
                    const event_vector& deps = {});

/// Partially shuffles the first `top` elements of an array using the Fisher-Yates algorithm.
/// @tparam Type The data type of the array elements.
/// @param[in] queue_ The SYCL queue for device execution.
/// @param[in, out] result_array The array to be partially shuffled.
/// @param[in] top The number of elements to shuffle.
/// @param[in] seed The seed for the engine.
/// @param[in] method The rng engine type. Defaults to `mt19937`.
/// @param[in] deps Dependencies for the SYCL event.
template <typename Type>
sycl::event partial_fisher_yates_shuffle(
    sycl::queue& queue_,
    ndview<Type, 1>& result_array,
    std::int64_t top,
    std::int64_t seed,
    engine_type_internal method = engine_type_internal::mt19937,
    const event_vector& deps = {});
#endif

} // namespace oneapi::dal::backend::primitives
