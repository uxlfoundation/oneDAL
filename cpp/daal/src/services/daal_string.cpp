/** file daal_string.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include <cstring>

#include "services/daal_string.h"
#include "src/services/service_defines.h"
#include "src/externals/service_service.h"

namespace daal
{
namespace services
{
DAAL_EXPORT const int String::__DAAL_STR_MAX_SIZE = DAAL_MAX_STRING_SIZE;

void String::initialize(const char * str, const size_t length)
{
    if (length)
    {
        _c_str = (char *)daal::services::daal_calloc(sizeof(char) * (length + 1));
        if (!_c_str)
        {
            return;
        }

        daal::internal::ServiceInst::serv_strncpy_s(_c_str, length + 1, str, length + 1);
    }
}

void String::reset()
{
    if (_c_str)
    {
        daal_free(_c_str);
    }
}

String::String() : _c_str(0) {}

String::String(size_t length, char placeholder) : _c_str(0)
{
    if (length)
    {
        _c_str = (char *)daal::services::daal_calloc(sizeof(char) * (length + 1));
        if (!_c_str)
        {
            return;
        }

        for (size_t i = 0; i < length; i++)
        {
            _c_str[i] = placeholder;
        }
        _c_str[length] = '\0';
    }
}

String::String(const char * begin, const char * end) : _c_str(0)
{
    initialize(begin, end - begin);
}

String::String(const char * str, size_t capacity) : _c_str(0)
{
    size_t strLength = 0;
    if (str)
    {
        strLength = daal::internal::ServiceInst::serv_strnlen_s(str, String::__DAAL_STR_MAX_SIZE);
    }
    initialize(str, strLength);
};

String::String(const String & str) : _c_str(0)
{
    initialize(str.c_str(), str.length());
};

String::~String()
{
    reset();
}

String & String::operator=(const String & other)
{
    if (this != &other)
    {
        reset();
        initialize(other.c_str(), other.length());
    }
    return *this;
}

bool String::operator==(const String & other)
{
    if (this == &other) return true;
    if (this->length() != other.length()) return false;
    return strncmp(this->c_str(), other.c_str(), this->length()) == 0;
}

bool String::operator!=(const String & other)
{
    return !(*this == other);
}

size_t String::length() const
{
    if (_c_str)
    {
        return daal::internal::ServiceInst::serv_strnlen_s(_c_str, String::__DAAL_STR_MAX_SIZE);
    }
    return 0;
}

void String::add(const String & str)
{
    size_t prevLength = length();
    char * prevStr    = (char *)daal::services::daal_calloc(sizeof(char) * (prevLength + 1));
    daal::internal::ServiceInst::serv_strncpy_s(prevStr, prevLength + 1, _c_str, prevLength + 1);

    size_t newLength = prevLength + str.length() + 1;
    if (_c_str)
    {
        daal_free(_c_str);
    }
    _c_str = (char *)daal::services::daal_calloc(sizeof(char) * (newLength + 1));

    daal::internal::ServiceInst::serv_strncpy_s(_c_str, prevLength + 1, prevStr, prevLength + 1);
    daal::internal::ServiceInst::serv_strncat_s(_c_str, newLength, str.c_str(), newLength - prevLength);

    if (prevStr)
    {
        daal_free(prevStr);
    }
}

String & String::operator+(const String & str)
{
    add(str);
    return *this;
}

char String::operator[](size_t index) const
{
    return _c_str[index];
}

char String::get(size_t index) const
{
    return _c_str[index];
}

const char * String::c_str() const
{
    return _c_str;
}

} // namespace services
} // namespace daal
