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

#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/spmd/common.hpp"

namespace oneapi::dal::detail {
namespace v1 {

struct ops_error_handling_dispatcher {
    template <typename Policy, typename Ops>
    auto operator()(Policy&& policy, Ops&& op) {
        using result_t = decltype(op());
        if constexpr (is_distributed_policy_v<std::decay_t<Policy>>) {
            try {
                try {
                    return op();
                }
                catch (const dal::preview::spmd::error_holder& e) {
                    throw e;
                }
                catch (...) {
                    policy.get_communicator().set_active_exception(std::current_exception());
                }
                policy.get_communicator().wait_for_exception_handling();
            }
            catch (const dal::preview::spmd::error_holder& e) {
                e.rethrow_actual();
            }
        }
        else {
            return op();
        }

        return result_t{};
    }
};

template <typename T, typename Enable = void>
struct is_parametrized : std::false_type {};

template <typename T>
struct is_parametrized<T, enable_if_type_t<typename T::param_t>> : std::true_type {};

template <typename T>
constexpr bool is_parametrized_v = is_parametrized<T>::value;

template <typename T, typename Ops>
constexpr bool is_ops_parameter_v = std::is_same_v<T, typename Ops::param_t>;

template <typename T, typename Ops>
constexpr bool is_input_v = std::is_same_v<T, typename Ops::input_t>;

template <typename T, typename Ops>
constexpr bool is_parameter() {
    if constexpr (is_parametrized_v<Ops>) {
        return is_ops_parameter_v<T, Ops>;
    }
    else {
        return false;
    }
}

template <typename T, typename Ops>
constexpr bool is_parameter_v = is_parameter<T, Ops>();

template <typename T, typename Ops, bool IsParameter = false, bool IsInput = is_input_v<T, Ops>>
struct ops_input_dispatcher {
    template <typename... Args>
    auto operator()(Args&&... args) {
        return Ops{}(std::forward<Args>(args)...);
    }
};

template <typename T, typename Ops>
struct ops_input_dispatcher<T, Ops, /*IsParameter = */ false, /* IsInput = */ false> {
    template <typename Policy, typename Descriptor, typename... Args>
    auto operator()(Policy&& policy, Descriptor&& desc, Args&&... args) {
        using input_t = typename Ops::input_t;

        return ops_error_handling_dispatcher{}(std::forward<Policy>(policy), [&]() {
            return Ops{}(std::forward<Policy>(policy),
                         std::forward<Descriptor>(desc),
                         input_t{ std::forward<Args>(args)... });
        });
    }
};

template <typename T, typename Ops>
struct ops_input_dispatcher<T, Ops, /*IsParameter = */ true, /* IsInput = */ false> {
    template <typename Policy, typename Descriptor, typename Parameter, typename... Args>
    auto operator()(Policy&& policy, Descriptor&& desc, Parameter&& param, Args&&... args) {
        using input_t = typename Ops::input_t;

        return ops_error_handling_dispatcher{}(std::forward<Policy>(policy), [&]() {
            return Ops{}(std::forward<Policy>(policy),
                         std::forward<Descriptor>(desc),
                         std::forward<Parameter>(param),
                         input_t{ std::forward<Args>(args)... });
        });
    }
};

template <typename T, typename Ops, bool IsParameter = is_parameter_v<T, Ops>>
struct ops_parameter_dispatcher;

template <typename T, typename Ops>
struct ops_parameter_dispatcher<T, Ops, /* IsParameter = */ true> {
    template <typename Policy,
              typename Descriptor,
              typename Parameter,
              typename Head,
              typename... Tail>
    auto operator()(Policy&& policy,
                    Descriptor&& desc,
                    Parameter&& param,
                    Head&& head,
                    Tail&&... tail) {
        using dispatcher_t = ops_input_dispatcher<std::decay_t<Head>, Ops, true>;
        return dispatcher_t{}(std::forward<Policy>(policy),
                              std::forward<Descriptor>(desc),
                              std::forward<Parameter>(param),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

template <typename T, typename Ops>
struct ops_parameter_dispatcher<T, Ops, /* IsParameter = */ false> {
    template <typename Policy, typename Descriptor, typename Head, typename... Tail>
    auto operator()(Policy&& policy, Descriptor&& desc, Head&& head, Tail&&... tail) {
        using dispatcher_t = ops_input_dispatcher<std::decay_t<Head>, Ops, false>;
        return dispatcher_t{}(std::forward<Policy>(policy),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

template <typename T, template <typename> typename Ops, bool IsPolicy = is_execution_policy_v<T>>
struct ops_policy_dispatcher;

template <typename T, template <typename> typename Ops>
struct ops_policy_dispatcher<T, Ops, /* IsPolicy = */ true> {
    template <typename Policy, typename Descriptor, typename Head, typename... Tail>
    auto operator()(Policy&& policy, Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Descriptor>>;
        using dispatcher_t = ops_parameter_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(std::forward<Policy>(policy),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

template <typename T, template <typename> typename Ops>
struct ops_policy_dispatcher<T, Ops, /* IsPolicy = */ false> {
    template <typename Descriptor, typename Head, typename... Tail>
    auto operator()(Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Descriptor>>;
        using dispatcher_t = ops_parameter_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(host_policy::get_default(),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

template <typename Object,
          typename T,
          template <typename, typename> typename Ops,
          bool IsPolicy = is_execution_policy_v<T>>
struct ops_policy_dispatcher_object;

template <typename Object, typename T, template <typename, typename> typename Ops>
struct ops_policy_dispatcher_object<Object, T, Ops, /* IsPolicy = */ true> {
    template <typename Policy, typename Descriptor>
    auto operator()(Policy&& policy, Descriptor&& desc) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>>;
        using input_t = typename ops_t::input_t;
        return ops_error_handling_dispatcher{}(std::forward<Policy>(policy), [&]() {
            return ops_t{}(std::forward<Policy>(policy), std::forward<Descriptor>(desc), input_t{});
        });
    }

    template <typename Policy, typename Descriptor, typename Head, typename... Tail>
    auto operator()(Policy&& policy, Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>>;
        using dispatcher_t = ops_input_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(std::forward<Policy>(policy),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

template <typename Object, typename T, template <typename, typename> typename Ops>
struct ops_policy_dispatcher_object<Object, T, Ops, /* IsPolicy = */ false> {
    template <typename Descriptor>
    auto operator()(Descriptor&& desc) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>>;
        using input_t = typename ops_t::input_t;
        return ops_t{}(host_policy::get_default(), std::forward<Descriptor>(desc), input_t{});
    }

    template <typename Descriptor, typename Head, typename... Tail>
    auto operator()(Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>>;
        using dispatcher_t = ops_input_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(host_policy::get_default(),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

} // namespace v1

using v1::ops_input_dispatcher;
using v1::ops_policy_dispatcher;
using v1::ops_policy_dispatcher_object;

} // namespace oneapi::dal::detail
