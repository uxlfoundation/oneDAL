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

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

// Value of the command-line option `name` ("--rows 100" or "--rows=100"), or `fallback`.
// An option given without a value exits with a message.
inline std::string get_option(int argc,
                              char const* argv[],
                              const char* name,
                              const std::string& fallback) {
    const std::size_t length = std::strlen(name);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            if (i + 1 == argc || std::strncmp(argv[i + 1], "--", 2) == 0) {
                std::cerr << name << " expects a value" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            return argv[i + 1];
        }
        if (std::strncmp(argv[i], name, length) == 0 && argv[i][length] == '=') {
            return argv[i] + length + 1;
        }
    }
    return fallback;
}

// Value of the positive integer option `name`, or `fallback`; any other value exits with a message.
inline std::int64_t get_option(int argc,
                               char const* argv[],
                               const char* name,
                               std::int64_t fallback) {
    const std::string value = get_option(argc, argv, name, std::string{});
    if (value.empty()) {
        return fallback;
    }
    char* end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(value.c_str(), &end, 10);
    if (errno != 0 || *end != '\0' || parsed <= 0) {
        std::cerr << name << " expects a positive integer, got '" << value << "'" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return parsed;
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
