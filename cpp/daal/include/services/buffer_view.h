/* file: buffer_view.h */
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
//  Implementation of a dummy base class needed to fix ABI inconsistency between
//  Visual Studio* 2012 and 2013.
//--
*/

#ifndef __DAAL_SERVICES_BUFFER_VIEW_H__
#define __DAAL_SERVICES_BUFFER_VIEW_H__

#include "services/daal_defines.h"
#include "services/collection.h"

namespace daal
{
namespace services
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup memory
 * @{
 */

/**
 *  <a name="DAAL-CLASS-SERVICES__BUFFERVIEW"></a>
 *  \brief   Class that refers to a contiguous sequence of objects, but
 *           doesn't control allocated memory buffer and objects lifetime,
 *           user is responsible for correct memory management and deallocation
 *  \tparam  T  Type of an object stored in the buffer
 */
template <typename T>
class BufferView : public Base
{
public:
    /**
     * Creates empty BufferView
     */
    BufferView() : _buffer(NULL), _size(0) {}

    /**
     * Creates BufferView from the raw data
     * \param[in]  buffer      The raw pointer to the buffer
     * \param[in]  bufferSize  The buffer size
     */
    explicit BufferView(T * buffer, size_t bufferSize) : _buffer(buffer), _size(bufferSize) { DAAL_ASSERT(_buffer); }

    /**
     *  Returns pointer to the underlying buffer serving as element storage
     *  \return Pointer to the array
     */
    T * data() const { return _buffer; }

    /**
     *  Size of a buffer
     *  \return Size of the buffer
     */
    size_t size() const { return _size; }

    /**
     *  Flag indicates that buffer is empty (its size is 0)
     *  \return Whether the buffer is empty
     */
    bool empty() const { return (_buffer == NULL) || (_size == 0); }

    /**
    *  Const element access
    *  \param[in] index Index of an accessed element
    *  \return    Const reference to the element
    */
    const T & operator[](size_t index) const
    {
        DAAL_ASSERT(index < _size);
        return _buffer[index];
    }

    /**
     *  Element access
     *  \param[in] index Index of an accessed element
     *  \return    Reference to the element
     */
    T & operator[](size_t index)
    {
        DAAL_ASSERT(index < _size);
        return _buffer[index];
    }

    /**
     * Gets the block of the current buffer
     * \param[in]  offset  The offset of the block
     * \param[in]  size    The size of the block
     * \return     New BufferView object for specified block
     */
    BufferView getBlock(size_t offset, size_t size) const
    {
        DAAL_ASSERT(offset + size <= _size);
        return BufferView<T>(_buffer + offset, size);
    }

private:
    T * _buffer;
    size_t _size;
};

/** @} */
} // namespace interface1

using interface1::BufferView;

} // namespace services
} // namespace daal

#endif
