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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

// Value of the command-line option `name` ("--rows 100" or "--rows=100"), or `fallback`.
inline std::string get_option(int argc,
                              char const* argv[],
                              const char* name,
                              const std::string& fallback) {
    const std::size_t length = std::strlen(name);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
            return argv[i + 1];
        }
        if (std::strncmp(argv[i], name, length) == 0 && argv[i][length] == '=') {
            return argv[i] + length + 1;
        }
    }
    return fallback;
}

inline std::int64_t get_option(int argc,
                               char const* argv[],
                               const char* name,
                               std::int64_t fallback) {
    const std::string value = get_option(argc, argv, name, std::string{});
    return value.empty() ? fallback : std::atoll(value.c_str());
}

// Whether the flag `name` is present, as in "--help".
inline bool has_flag(int argc, char const* argv[], const char* name) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return true;
        }
    }
    return false;
}
