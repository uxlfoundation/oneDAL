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

#include "oneapi/dal/io/csv/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::csv::detail {
namespace v1 {

class data_source_impl : public base {
public:
    char delimiter = ',';
    bool parse_header = false;
    std::string file_name = "";
};

data_source_base::data_source_base(const char* file_name) : impl_(new data_source_impl{}) {
    set_file_name_impl(file_name);
}

char data_source_base::get_delimiter_impl() const {
    return impl_->delimiter;
}

bool data_source_base::get_parse_header_impl() const {
    return impl_->parse_header;
}

const char* data_source_base::get_file_name_impl() const {
    return impl_->file_name.c_str();
}

void data_source_base::set_delimiter_impl(char value) {
    impl_->delimiter = value;
}

void data_source_base::set_parse_header_impl(bool value) {
    impl_->parse_header = value;
}

void data_source_base::set_file_name_impl(const char* value) {
    impl_->file_name = std::string(value);
}

} // namespace v1
} // namespace oneapi::dal::csv::detail
