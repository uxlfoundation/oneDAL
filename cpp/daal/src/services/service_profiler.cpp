/* file: service_profiler.cpp */
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

/*
//++
//  Profiler for time measurement of kernels
//--
*/

#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <algorithm>
#include "src/services/service_profiler.h"

namespace daal
{
namespace internal
{

static void set_verbose_from_env()
{
    const char * verbose_str = std::getenv("ONEDAL_VERBOSE");
    int newval               = PROFILER_MODE_OFF;
    if (verbose_str)
    {
        char * endptr  = nullptr;
        errno          = 0;
        long val       = std::strtol(verbose_str, &endptr, 10);
        bool parsed_ok = (errno == 0 && endptr != verbose_str && *endptr == '\0');
        if (parsed_ok && val >= 0 && val <= 5) newval = static_cast<int>(val);
    }
    daal_verbose_val = newval;
}

static int daal_verbose_mode()
{
    if (daal_verbose_val == -1) set_verbose_from_env();
    return daal_verbose_val;
}

static std::string format_time_for_output(std::uint64_t time_ns)
{
    std::ostringstream out;
    double time = static_cast<double>(time_ns);
    if (time <= 0)
        out << "0.00s";
    else if (time > 1e9)
        out << std::fixed << std::setprecision(2) << time / 1e9 << "s";
    else if (time > 1e6)
        out << std::fixed << std::setprecision(2) << time / 1e6 << "ms";
    else if (time > 1e3)
        out << std::fixed << std::setprecision(2) << time / 1e3 << "us";
    else
        out << static_cast<std::uint64_t>(time) << "ns";
    return out.str();
}

static bool is_service_debug_enabled()
{
    static const bool service_debug_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_DEBUG;
    }();
    return service_debug_value;
}

static bool is_logger_enabled()
{
    static const bool logger_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_LOGGER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return logger_value;
}

static bool is_tracer_enabled()
{
    static const bool verbose_value = [] {
        std::ios::sync_with_stdio(false);
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_TRACER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return verbose_value;
}
static bool is_profiler_enabled()
{
    static const bool profiler_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_LOGGER || value == PROFILER_MODE_TRACER || value == PROFILER_MODE_ANALYZER || value == PROFILER_MODE_ALL_TOOLS
               || value == PROFILER_MODE_DEBUG;
    }();
    return profiler_value;
}

static bool is_analyzer_enabled()
{
    static const bool profiler_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_ANALYZER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return profiler_value;
}

void print_header()
{
    if (is_profiler_enabled())
    {
        daal::services::LibraryVersionInfo ver;
        std::cerr << "Major version:          " << ver.majorVersion << '\n';
        std::cerr << "Minor version:          " << ver.minorVersion << '\n';
        std::cerr << "Update version:         " << ver.updateVersion << '\n';
        std::cerr << "Product status:         " << ver.productStatus << '\n';
        std::cerr << "Build:                  " << ver.build << '\n';
        std::cerr << "Build revision:         " << ver.build_rev << '\n';
        std::cerr << "Name:                   " << ver.name << '\n';
        std::cerr << "Processor optimization: " << ver.processor << '\n';
        std::cerr << '\n';
    }
}

} // namespace internal
} // namespace daal