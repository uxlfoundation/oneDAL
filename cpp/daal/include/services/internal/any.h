/* file: any.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_ANY_H__
#define __DAAL_SERVICES_INTERNAL_ANY_H__

namespace daal
{
namespace services
{
namespace internal
{
namespace interface1
{
/** @ingroup services_internal
 * @{
 */

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__ANY"></a>
 *  \brief Class-container for any value
 */
class Any : public Base
{
private:
    class AbstractValue : public Base
    {
    public:
        virtual AbstractValue * copy() const = 0;
    };

    template <typename T>
    class Value : public AbstractValue
    {
    public:
        explicit Value(const T & value) : _value(value) {}

        const T & get() const { return _value; }

        T & get() { return _value; }

        Value<T> * copy() const override { return new Value<T>(_value); }

    private:
        T _value;
    };

public:
    Any() : _value(NULL) {}

    template <typename T>
    Any(const T & value) : _value(new Value<T>(value))
    {}

    Any(const Any & other) : _value(other._value ? other._value->copy() : NULL) {}

    ~Any() override { delete _value; }

    bool empty() const { return _value == NULL; }

    template <typename T>
    bool check() const
    {
        return dynamic_cast<const Value<T> *>(_value) != NULL;
    }

    template <typename T>
    const T & get() const
    {
        return static_cast<const Value<T> *>(_value)->get();
    }

    template <typename T>
    T & get()
    {
        return static_cast<Value<T> *>(_value)->get();
    }

    Any & swap(Any & other)
    {
        AbstractValue * tmp = _value;
        _value              = other._value;
        other._value        = tmp;
        return *this;
    }

    template <typename T>
    Any & operator=(const T & value)
    {
        delete _value;
        _value = new Value<T>(value);
        return *this;
    }

    Any & operator=(Any other) { return swap(other); }

private:
    AbstractValue * _value;
};

/** @} */
} // namespace interface1

using interface1::Any;

} // namespace internal
} // namespace services
} // namespace daal

#endif
