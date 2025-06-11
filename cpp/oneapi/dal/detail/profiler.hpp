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

#include "src/services/service_profiler.h"

// UTILS
#define ONEDAL_PROFILER_MACRO_1(name)                       ONEDAL_PROFILER_START_TASK(name)
#define ONEDAL_PROFILER_MACRO_2(name, queue)                ONEDAL_PROFILER_START_TASK_WITH_QUEUE(name, queue)
#define ONEDAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

// START_TASKS
#define ONEDAL_PROFILER_START_TASK(name) daal::internal::profiler::start_task(#name)
#define ONEDAL_PROFILER_START_TASK_WITH_QUEUE(name, queue) \
    daal::internal::profiler::start_task(#name)
#define ONEDAL_PROFILER_START_NULL_TASK() daal::internal::profiler::start_task(nullptr)

// PROFILER TASKS
#define ONEDAL_PROFILER_TASK_WITH_ARGS(task_name, ...)                                     \
    daal::internal::profiler_task __profiler_task =                                        \
        (daal::internal::is_profiler_enabled()) ? [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_logger_enabled()) {                                         \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                \
        }                                                                                  \
        return ONEDAL_PROFILER_START_TASK(task_name);                                      \
    }()                                                                                    \
        : ONEDAL_PROFILER_START_NULL_TASK()

#define ONEDAL_PROFILER_TASK_WITH_ARGS_QUEUE(task_name, queue, ...)                        \
    daal::internal::profiler_task __profiler_task =                                        \
        (daal::internal::is_profiler_enabled()) ? [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_logger_enabled()) {                                         \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                \
        }                                                                                  \
        return ONEDAL_PROFILER_START_TASK_WITH_QUEUE(task_name, queue);                    \
    }()                                                                                    \
        : ONEDAL_PROFILER_START_NULL_TASK()

#define ONEDAL_PROFILER_TASK(...)                                                          \
    daal::internal::profiler_task __profiler_task =                                        \
        (daal::internal::is_profiler_enabled()) ? [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_logger_enabled()) {                                         \
            DAAL_PROFILER_PRINT_HEADER();                                                  \
            std::cerr << "Profiler task_name: " << #__VA_ARGS__ << '\n';                   \
        }                                                                                  \
        return ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                                      \
                                         ONEDAL_PROFILER_MACRO_2,                          \
                                         ONEDAL_PROFILER_MACRO_1,                          \
                                         FICTIVE)(__VA_ARGS__);                            \
    }()                                                                                    \
        : ONEDAL_PROFILER_START_NULL_TASK()

#define ONEDAL_PROFILER_SERVICE_TASK_WITH_ARGS(task_name, ...)                                  \
    daal::internal::profiler_task __profiler_task =                                             \
        (daal::internal::is_service_debug_enabled()) ? [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_logger_enabled()) {                                              \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                     \
        }                                                                                       \
        return ONEDAL_PROFILER_START_TASK(task_name);                                           \
    }()                                                                                         \
        : ONEDAL_PROFILER_START_NULL_TASK()

#define ONEDAL_PROFILER_SERVICE_TASK_WITH_ARGS_QUEUE(task_name, queue, ...)                     \
    daal::internal::profiler_task __profiler_task =                                             \
        (daal::internal::is_service_debug_enabled()) ? [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_logger_enabled()) {                                              \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                     \
        }                                                                                       \
        return ONEDAL_PROFILER_START_TASK_WITH_QUEUE(task_name, queue);                         \
    }()                                                                                         \
        : ONEDAL_PROFILER_START_NULL_TASK()

#define ONEDAL_PROFILER_SERVICE_TASK(...)                                                       \
    daal::internal::profiler_task __profiler_task =                                             \
        (daal::internal::is_service_debug_enabled()) ? [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_logger_enabled()) {                                              \
            DAAL_PROFILER_PRINT_HEADER();                                                       \
            std::cerr << "Profiler task_name: " << #__VA_ARGS__ << '\n';                        \
        }                                                                                       \
        return ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                                           \
                                         ONEDAL_PROFILER_MACRO_2,                               \
                                         ONEDAL_PROFILER_MACRO_1,                               \
                                         FICTIVE)(__VA_ARGS__);                                 \
    }()                                                                                         \
        : ONEDAL_PROFILER_START_NULL_TASK()
