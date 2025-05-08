/** file daal_string.h */
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

/*
//++
//  oneAPI Data Analytics Library (oneDAL) string class.
//--
*/

#ifndef __DAAL_STRING__
#define __DAAL_STRING__

#include <string>
#include "services/base.h"

namespace daal
{
namespace services
{
namespace interface1
{
/**
 * @ingroup memory
 * @{
 */

/**
 * <a name="DAAL-CLASS-SERVICES__STRING"></a>
 * \brief Class that implements functionality of the string,
 *        an object that represents a sequence of characters
 */
class DAAL_EXPORT String : public Base
{
public:
    /**
     * Default constructor
     */
    String();

    /**
     * Constructs string of specified length filled with \p placeholder
     * \param[in]  length       The length of the string
     * \param[in]  placeholder  The character to be used as placeholder
     */
    explicit String(size_t length, char placeholder = '\0');

    /**
     * Range constructor
     * \param[in]  begin  Pointer to the first character of the string
     * \param[in]  end    Pointer to the last character + 1
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED explicit String(const char * begin, const char * end);

    /**
     * Constructor from STL string
     * \param[in]  str The STL string
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED String(const std::string & str) { initialize(str.c_str(), str.size()); }

    /**
     * Default constructor
     * \param[in] str       The sequence of characters that forms the string
     * \param[in] capacity  Unused
     *
     */
    String(const char * str, size_t capacity = 0);

    /**
     * Copy constructor
     * \param[in] str       The sequence of characters that forms the string
     */
    String(const String & str);

    /**
     * Destructor
     */
    ~String();

    /**
     * Assigment operator
     */
    String & operator=(const String & other);

    /**
     * Comparison operators: equal to
     */
    bool operator==(const String & other);

    /**
     * Comparison operators: not equal to
     */
    bool operator!=(const String & other);

    /**
     * Returns the number of characters in the string
     * \return The number of characters in the string
     */
    size_t length() const;

    /**
     * Extends the string by appending additional characters at the end of its current value
     * \param[in] str A string object whose values are copied at the end
     */
    void add(const String & str);

    /**
     * Extends the string by appending additional characters at the end of its current value
     * \param[in] str A string object whose values are copied at the end
     */
    String & operator+(const String & str);

    /**
     * Returns the pointer to a character of the string
     * \param[in] index     Index of the character
     * \return  Pointer to the character of the string
     */
    char operator[](size_t index) const;

    /**
     * Returns the pointer to a character of the string
     * \param[in] index     Index of the character
     * \return  Pointer to the character of the string
     */
    char get(size_t index) const;

    /**
     * Returns the content of the string as array of characters
     * \return The content of the string as array of characters
     */
    const char * c_str() const;

    static const int __DAAL_STR_MAX_SIZE; /*!< Maximal length of the string */

private:
    char * _c_str;

    void reset();

    void initialize(const char * str, const size_t length);
};
/** @} */

/**
 * @ingroup memory
 * @{
 */

/**
 * <a name="DAAL-CLASS-SERVICES__STRINGVIEW"></a>
 * \brief Class that implements functionality of the string but doesn't manage provided
 *        memory, user is responsible for correct memory management and deallocation
 */
class StringView : public Base
{
public:
    /**
     * Creates empty StringView
     */
    StringView() : _cStr(NULL), _length(0) {}

    /**
     * Creates StringView from the raw C-style string
     * \param[in]  cstr    The sequence of characters that forms the string
     * \param[in]  length  The length of string except termination character
     */
    explicit StringView(const char * cstr, size_t length) : _cStr(cstr), _length(length) {}

    /**
     * Returns the content of the string as array of characters
     * \return The content of the string as array of characters
     */
    const char * c_str() const { return _cStr; }

    /**
     * Returns the number of characters in the string
     * \return The number of characters in the string
     */
    size_t size() const { return _length; }

    /**
     *  Flag indicates that string is empty (its size is 0)
     *  \return Whether the string is empty
     */
    bool empty() const { return (_cStr == NULL) || (_length == 0); }

    /**
     * Returns the character of the string
     * \param[in] index The index of the character
     * \return The character of the string
     */
    char operator[](size_t index) const
    {
        DAAL_ASSERT(index < _length);
        return _cStr[index];
    }

    /**
     * Returns pointer to the first character of the string
     * \return The constant pointer to the first character of the string
     */
    const char * begin() const { return _cStr; }

    /**
     * Returns pointer to the last + 1 character of the string
     * \return The constant pointer to the last + 1 character of the string
     */
    const char * end() const { return _cStr + _length; }

private:
    const char * _cStr;
    size_t _length;
};
/** @} */

} // namespace interface1

using interface1::String;
using interface1::StringView;

} // namespace services
} // namespace daal

#endif
