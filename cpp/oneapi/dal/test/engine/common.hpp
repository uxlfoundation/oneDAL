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

#pragma once

#include <tuple>
#include <memory>
#include <iostream>
#include <type_traits>

#include <fmt/core.h>
//Necessary headers from boost
#include <boost/process.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "oneapi/dal/train.hpp"
#include "oneapi/dal/infer.hpp"
#include "oneapi/dal/compute.hpp"
#include "oneapi/dal/partial_compute.hpp"
#include "oneapi/dal/finalize_compute.hpp"
#include "oneapi/dal/partial_train.hpp"
#include "oneapi/dal/finalize_train.hpp"
#include "oneapi/dal/exceptions.hpp"

#include "oneapi/dal/test/engine/catch.hpp"
#include "oneapi/dal/test/engine/config.hpp"
#include "oneapi/dal/test/engine/macro.hpp"
#include "oneapi/dal/test/engine/type_traits.hpp"

// Workaround DPC++ Compiler's warning on unused
// variable declared by Catch2's TEST_CASE macro
#ifdef __clang__
#define _TE_DISABLE_UNUSED_VARIABLE \
    _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wunused-variable\"")

#define _TE_ENABLE_UNUSED_VARIABLE _Pragma("clang diagnostic pop")

#undef TEST_CASE
#define TEST_CASE(...)                   \
    _TE_DISABLE_UNUSED_VARIABLE          \
    INTERNAL_CATCH_TESTCASE(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_TEST_CASE
#define TEMPLATE_TEST_CASE(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                    \
    INTERNAL_CATCH_TEMPLATE_TEST_CASE(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_LIST_TEST_CASE
#define TEMPLATE_LIST_TEST_CASE(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                         \
    INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_TEST_CASE_SIG
#define TEMPLATE_TEST_CASE_SIG(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                        \
    INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEST_CASE_METHOD
#define TEST_CASE_METHOD(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                  \
    INTERNAL_CATCH_TEST_CASE_METHOD(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_TEST_CASE_METHOD
#define TEMPLATE_TEST_CASE_METHOD(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                           \
    INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_LIST_TEST_CASE_METHOD
#define TEMPLATE_LIST_TEST_CASE_METHOD(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                                \
    INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_METHOD(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_TEST_CASE_METHOD_SIG
#define TEMPLATE_TEST_CASE_METHOD_SIG(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                               \
    INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE
#endif // __clang__

// Shortcuts for Catch2 defines
#define TEST                 TEST_CASE
#define TEMPLATE_TEST        TEMPLATE_TEST_CASE
#define TEMPLATE_LIST_TEST   TEMPLATE_LIST_TEST_CASE
#define TEMPLATE_SIG_TEST    TEMPLATE_TEST_CASE_SIG
#define TEST_M               TEST_CASE_METHOD
#define TEMPLATE_TEST_M      TEMPLATE_TEST_CASE_METHOD
#define TEMPLATE_LIST_TEST_M TEMPLATE_LIST_TEST_CASE_METHOD
#define TEMPLATE_SIG_TEST_M  TEMPLATE_TEST_CASE_METHOD_SIG

#ifdef ONEDAL_DATA_PARALLEL
#define DECLARE_TEST_POLICY(policy_name) oneapi::dal::test::engine::device_test_policy policy_name
#else
#define DECLARE_TEST_POLICY(policy_name) oneapi::dal::test::engine::host_test_policy policy_name
#endif

#define SKIP_IF(condition) \
    if (condition) {       \
        SUCCEED();         \
        return;            \
    }

#define REGISTER_GLOBAL_SETUP(unique_id, GlobalSetup)                                      \
    namespace {                                                                            \
    int global_setup_init() {                                                              \
        ::oneapi::dal::test::engine::register_global_setup(#unique_id, new GlobalSetup{}); \
        return 0;                                                                          \
    }                                                                                      \
    [[maybe_unused]] const volatile int _ = global_setup_init();                           \
    }

namespace oneapi::dal::test::engine {

template <typename T>
struct type2str {
    static const char* name() {
        return "Unknown";
    }
};

#define SPECIALIZE_TYPE_MAP(T)     \
    template <>                    \
    struct type2str<T> {           \
        static const char* name(); \
    };

SPECIALIZE_TYPE_MAP(float)
SPECIALIZE_TYPE_MAP(double)
SPECIALIZE_TYPE_MAP(std::uint8_t)
SPECIALIZE_TYPE_MAP(std::uint16_t)
SPECIALIZE_TYPE_MAP(std::uint32_t)
SPECIALIZE_TYPE_MAP(std::uint64_t)
SPECIALIZE_TYPE_MAP(std::int8_t)
SPECIALIZE_TYPE_MAP(std::int16_t)
SPECIALIZE_TYPE_MAP(std::int32_t)
SPECIALIZE_TYPE_MAP(std::int64_t)

#undef SPECIALIZE_TYPE_MAP

class global_setup_action {
public:
    virtual ~global_setup_action() = default;
    virtual void init(const global_config& config) = 0;
    virtual void tear_down() = 0;
};

void register_global_setup(const std::string& name, global_setup_action* action);

void init_global_setup_actions(const global_config& config);

void tear_down_global_setup_actions();

class host_test_policy {
public:
    bool is_cpu() const {
        return true;
    }

    bool is_gpu() const {
        return false;
    }

    bool has_native_float64() const {
        return true;
    }
};

template <typename... Args>
inline auto train(host_test_policy& policy, Args&&... args) {
    return dal::train(std::forward<Args>(args)...);
}

template <typename... Args>
inline auto infer(host_test_policy& policy, Args&&... args) {
    return dal::infer(std::forward<Args>(args)...);
}

template <typename... Args>
inline auto compute(host_test_policy& policy, Args&&... args) {
    return dal::compute(std::forward<Args>(args)...);
}

template <typename... Args>
inline auto partial_compute(host_test_policy& policy, Args&&... args) {
    return dal::partial_compute(std::forward<Args>(args)...);
}

template <typename... Args>
inline auto finalize_compute(host_test_policy& policy, Args&&... args) {
    return dal::finalize_compute(std::forward<Args>(args)...);
}

template <typename... Args>
inline auto partial_train(host_test_policy& policy, Args&&... args) {
    return dal::partial_train(std::forward<Args>(args)...);
}

template <typename... Args>
inline auto finalize_train(host_test_policy& policy, Args&&... args) {
    return dal::finalize_train(std::forward<Args>(args)...);
}
#ifdef ONEDAL_DATA_PARALLEL
class test_queue_provider {
public:
    static test_queue_provider& get_instance();

    sycl::queue& get_global_queue() const {
        if (!queue_) {
            throw internal_error{ "Test queue provider is not initialized" };
        }
        return *queue_;
    }

    void init(const sycl::queue& queue) {
        queue_.reset(new sycl::queue{ queue });
    }

    void reset() {
        queue_.reset();
    }

private:
    test_queue_provider() = default;

    std::unique_ptr<sycl::queue> queue_;
};

inline const sycl::queue& get_global_queue() {
    return test_queue_provider::get_instance().get_global_queue();
}

class device_test_policy {
public:
    device_test_policy(const sycl::queue& queue) : queue_(queue) {}

    device_test_policy() : queue_(get_global_queue()) {}

    sycl::queue& get_queue() {
        return queue_;
    }

    bool is_cpu() const {
        return queue_.get_device().is_cpu();
    }

    bool is_gpu() const {
        return queue_.get_device().is_gpu();
    }

    bool has_native_float64() const;

private:
    sycl::queue queue_;
};

template <typename... Args>
inline auto train(device_test_policy& policy, Args&&... args) {
    return dal::train(policy.get_queue(), std::forward<Args>(args)...);
}

template <typename... Args>
inline auto infer(device_test_policy& policy, Args&&... args) {
    return dal::infer(policy.get_queue(), std::forward<Args>(args)...);
}

template <typename... Args>
inline auto compute(device_test_policy& policy, Args&&... args) {
    return dal::compute(policy.get_queue(), std::forward<Args>(args)...);
}

template <typename... Args>
inline auto partial_compute(device_test_policy& policy, Args&&... args) {
    return dal::partial_compute(policy.get_queue(), std::forward<Args>(args)...);
}

template <typename... Args>
inline auto finalize_compute(device_test_policy& policy, Args&&... args) {
    return dal::finalize_compute(policy.get_queue(), std::forward<Args>(args)...);
}

template <typename... Args>
inline auto partial_train(device_test_policy& policy, Args&&... args) {
    return dal::partial_train(policy.get_queue(), std::forward<Args>(args)...);
}

template <typename... Args>
inline auto finalize_train(device_test_policy& policy, Args&&... args) {
    return dal::finalize_train(policy.get_queue(), std::forward<Args>(args)...);
}
#endif

} // namespace oneapi::dal::test::engine
