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

#include "oneapi/dal/detail/array_impl.hpp"

namespace oneapi::dal {
namespace v2 {

/// @tparam T The type of the memory block elements within the array.
///              :literal:`T` can represent any type.
///
/// @pre :literal:`T` cannot be const-qualified.
template <typename T>
class array {
    static_assert(!std::is_const_v<T>, "array class cannot have const-qualified type of data");

    friend detail::pimpl_accessor;
    friend detail::serialization_accessor;

    template <typename U>
    friend class array;

    template <typename U>
    friend class chunked_array;

    friend class chunked_array_base;

    using impl_t = detail::array_impl<T>;

public:
    using data_t = T;

    /// Allocates a new memory block for mutable data, does not initialize it,
    /// creates a new array instance by passing a pointer to the memory block.
    /// The array owns the memory block (for details, see :txtref:`data_ownership_requirements`).
    ///
    /// @param count The number of elements of type :literal:`Data` to allocate memory for.
    /// @pre :expr:`count > 0`
    static array<T> empty(std::int64_t count) {
        return array<T>{
            impl_t::empty(detail::default_host_policy{}, count, detail::host_allocator<T>())
        };
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Allocates a new memory block for mutable data, does not initialize it,
    /// creates a new array instance by passing a pointer to the memory block.
    /// The array owns the memory block (for details, see :txtref:`data_ownership_requirements`).
    ///
    /// @param queue The SYCL* queue object.
    /// @param count The number of elements of type :literal:`T` to allocate memory for.
    /// @param alloc The kind of USM to be allocated.
    /// @pre :expr:`count > 0`
    static array<T> empty(const sycl::queue& queue,
                          std::int64_t count,
                          const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        return array<T>{ impl_t::empty(detail::data_parallel_policy{ queue },
                                       count,
                                       detail::data_parallel_allocator<T>(queue, alloc)) };
    }
#endif

    /// Allocates a new memory block for mutable data, fills it with a scalar value,
    /// creates a new array instance by passing a pointer to the memory block.
    /// The array owns the memory block (for details, see :txtref:`data_ownership_requirements`).
    ///
    /// @tparam Element The type from which array elements of type :literal:`T` can be constructed.
    ///
    /// @param count   The number of elements of type :literal:`T` to allocate memory for.
    /// @param element The value that is used to fill a memory block.
    /// @pre :expr:`count > 0`
    /// @pre Elements of type ``T`` are constructible from the ``Element`` type.
    template <typename K>
    static array<T> full(std::int64_t count, K&& element) {
        return array<T>{ impl_t::full(detail::default_host_policy{},
                                      count,
                                      std::forward<K>(element),
                                      detail::host_allocator<T>()) };
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Allocates a new memory block for mutable data, fills it with a scalar value,
    /// creates a new array instance by passing a pointer to the memory block.
    /// The array owns the memory block (for details, see :txtref:`data_ownership_requirements`).
    ///
    /// @tparam Element The type from which array elements of type :literal:`Data` can be constructed.
    ///
    /// @param queue   The SYCL* queue object.
    /// @param count   The number of elements of type :literal:`Data` to allocate memory for.
    /// @param element The value that is used to fill a memory block.
    /// @param alloc   The kind of USM to be allocated.
    /// @pre :expr:`count > 0`
    /// @pre Elements of type ``Data`` are constructible from the ``Element`` type.
    template <typename K>
    static array<T> full(sycl::queue& queue,
                         std::int64_t count,
                         K&& element,
                         const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        return array<T>{ impl_t::full(detail::data_parallel_policy{ queue },
                                      count,
                                      std::forward<K>(element),
                                      detail::data_parallel_allocator<T>(queue, alloc)) };
    }
#endif

    /// Allocates a new memory block on mutable data, fills it with zeros,
    /// creates a new array instance by passing a pointer to the memory block.
    /// The array owns the memory block (for details, see :txtref:`data_ownership_requirements`).
    ///
    /// @param count   The number of elements of type :literal:`Data` to allocate memory for.
    /// @pre :expr:`count > 0`
    static array<T> zeros(std::int64_t count) {
        // TODO: can be optimized in future
        return array<T>{
            impl_t::full(detail::default_host_policy{}, count, T{}, detail::host_allocator<T>())
        };
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Allocates a new memory block on mutable data, fills it with zeros,
    /// creates a new array instance by passing a pointer to the memory block.
    /// The array owns the memory block (for details, see :txtref:`data_ownership_requirements`).
    ///
    /// @param queue   The SYCL* queue object.
    /// @param count   The number of elements of type :literal:`T` to allocate memory for.
    /// @param alloc   The kind of USM to be allocated.
    /// @pre :expr:`count > 0`s
    static array<T> zeros(sycl::queue& queue,
                          std::int64_t count,
                          const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        // TODO: can be optimized in future
        return array<T>{ impl_t::full(dal::detail::data_parallel_policy{ queue },
                                      count,
                                      T{},
                                      detail::data_parallel_allocator<T>(queue, alloc)) };
    }
#endif

    /// Creates a new array instance by passing the pointer to externally-allocated memory block
    /// for mutable data. It is the responsibility of the calling application to free the memory block
    /// as the array does not free it when the reference count is zero.
    ///
    /// @param data         The pointer to externally-allocated memory block.
    /// @param count        The number of elements of type :literal:`Data` in the memory block.
    /// @pre :expr:`data != nullptr`
    /// @pre :expr:`count > 0`
    template <typename Y>
    static array<T> wrap(Y* data, std::int64_t count) {
        return array<T>{ data, count, dal::detail::empty_delete<const T>{} };
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Creates a new array instance by passing the pointer to externally-allocated memory block
    /// for mutable data. It is the responsibility of the calling application to free the memory block
    /// as the array does not free it when the reference count is zero.
    ///
    /// @param data         The pointer to externally-allocated memory block.
    /// @param count        The number of elements of type :literal:`Data` in the memory block.
    /// @param dependencies Events indicating availability of the :literal:`Data` for reading or writing.
    /// @pre :expr:`data != nullptr`
    /// @pre :expr:`count > 0`
    template <typename Y>
    [[deprecated]] static array<T> wrap(Y* data,
                                        std::int64_t count,
                                        const std::vector<sycl::event>& dependencies) {
        sycl::event::wait_and_throw(dependencies);
        return array<T>{ data, count, dal::detail::empty_delete<const T>{} };
    }
#endif

#ifdef ONEDAL_DATA_PARALLEL
    /// Creates a new array instance by passing the pointer to externally-allocated memory block
    /// for mutable data. It is the responsibility of the calling application to free the memory block
    /// as the array does not free it when the reference count is zero.
    ///
    /// @param queue        The SYCL* queue object.
    /// @param data         The pointer to externally-allocated memory block.
    /// @param count        The number of elements of type :literal:`T` in the memory block.
    /// @param dependencies Events indicating availability of the :literal:`Data` for reading or writing.
    /// @pre :expr:`data != nullptr`
    /// @pre :expr:`count > 0`
    template <typename Y>
    static array<T> wrap(const sycl::queue& queue,
                         Y* data,
                         std::int64_t count,
                         const std::vector<sycl::event>& dependencies = {}) {
        return array<T>{ queue, data, count, dal::detail::empty_delete<const T>{}, dependencies };
    }
#endif

    /// Creates a new instance of the class without memory allocation:
    /// :literal:`mutable_data` and :literal:`data` pointers should be set to ``nullptr``,
    /// :literal:`count` should be zero; the pointer to the ownership structure should be set to ``nullptr``.
    array() : impl_(new impl_t()) {
        reset_data();
    }

    /// Creates a new array instance that shares an ownership with :literal:`other` on its memory block.
    array(const array<T>& other) : impl_(new impl_t(*other.impl_)) {
        update_data(impl_.get());
    }

    /// Moves :literal:`data`, :literal:`mutable_data` pointers, :literal:`count`, and pointer to the ownership structure
    /// in :literal:`other` to the new array instance
    array(array<T>&& other) : impl_(std::move(other.impl_)) {
        update_data(impl_.get());
        other.reset_data();
    }

    // Add destructor
    ~array() = default;

    /// Creates a new array instance which owns a memory block of externally-allocated mutable data.
    /// The ownership structure is created for a block, the input :literal:`deleter`
    /// is assigned to it.
    ///
    /// @tparam Deleter     The type of a deleter used to free the :literal:`Data`.
    ///                     The deleter provides ``void operator()(Data*)`` member function.
    ///
    /// @param data         The pointer to externally-allocated memory block.
    /// @param count        The number of elements of type :literal:`Data` in the memory block.
    /// @param deleter      The object used to free :literal:`Data`.
    template <typename Deleter>
    explicit array(T* data, std::int64_t count, Deleter&& deleter)
            : impl_(new impl_t(detail::default_host_policy{},
                               data,
                               count,
                               std::forward<Deleter>(deleter))) {
        update_data(data, count);
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Creates a new array instance which owns a memory block of externally-allocated mutable data.
    /// The ownership structure is created for a block, the input :literal:`deleter`
    /// is assigned to it.
    ///
    /// @tparam Deleter     The type of a deleter used to free the :literal:`Data`.
    ///                     The deleter provides ``void operator()(Data*)`` member function.
    ///
    /// @param queue        The SYCL* queue object.
    /// @param data         The pointer to externally-allocated memory block.
    /// @param count        The number of elements of type :literal:`Data` in the memory block.
    /// @param deleter      The object used to free :literal:`Data`.
    template <typename Deleter>
    explicit array(const sycl::queue& queue,
                   T* data,
                   std::int64_t count,
                   Deleter&& deleter,
                   const std::vector<sycl::event>& dependencies = {})
            : impl_(new impl_t(detail::data_parallel_policy{ queue },
                               data,
                               count,
                               std::forward<Deleter>(deleter))) {
        ONEDAL_ASSERT(sycl::get_pointer_type(data, queue.get_context()) !=
                      sycl::usm::alloc::unknown);
        update_data(data, count);
        sycl::event::wait_and_throw(dependencies);
    }
#endif

    /// Creates a new array instance which owns a memory block of externally-allocated immutable data.
    /// The ownership structure is created for a block, the input :literal:`deleter`
    /// is assigned to it.
    ///
    /// @tparam ConstDeleter The type of a deleter used to free the :literal:`Data`.
    ///                      The deleter implements ``void operator()(const Data*)`` member function.
    ///
    /// @param data          The pointer to externally-allocated memory block.
    /// @param count         The number of elements of type :literal:`Data` in the :literal:`Data`.
    /// @param deleter       The object used to free :literal:`Data`.
    template <typename ConstDeleter>
    explicit array(const T* data, std::int64_t count, ConstDeleter&& deleter)
            : impl_(new impl_t(detail::default_host_policy{},
                               data,
                               count,
                               std::forward<ConstDeleter>(deleter))) {
        update_data(data, count);
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Creates a new array instance which owns a memory block of externally-allocated immutable data.
    /// The ownership structure is created for a block, the input :literal:`deleter`
    /// is assigned to it.
    ///
    /// @tparam ConstDeleter The type of a deleter used to free the :literal:`Data`.
    ///                      The deleter implements ``void operator()(const Data*)`` member function.
    ///
    /// @param queue         The SYCL* queue object.
    /// @param data          The pointer to externally-allocated memory block.
    /// @param count         The number of elements of type :literal:`Data` in the :literal:`Data`.
    /// @param deleter       The object used to free :literal:`Data`.
    /// @param dependencies  Events that indicate when :literal:`Data` becomes ready to be read or written
    template <typename ConstDeleter>
    explicit array(const sycl::queue& queue,
                   const T* data,
                   std::int64_t count,
                   ConstDeleter&& deleter,
                   const std::vector<sycl::event>& dependencies = {})
            : impl_(new impl_t(detail::data_parallel_policy{ queue },
                               data,
                               count,
                               std::forward<ConstDeleter>(deleter))) {
        ONEDAL_ASSERT(sycl::get_pointer_type(data, queue.get_context()) !=
                      sycl::usm::alloc::unknown);
        update_data(data, count);
        sycl::event::wait_and_throw(dependencies);
    }
#endif

    /// Creates a new array instance that shares ownership with the user-provided shared pointer.
    ///
    /// @param data         The shared pointer to externally-allocated memory block.
    /// @param count        The number of elements of type :literal:`Data` in the memory block.
    explicit array(const std::shared_ptr<T>& data, std::int64_t count)
            : impl_(new impl_t(detail::default_host_policy{}, data, count)) {
        update_data(data.get(), count);
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Creates a new array instance that shares ownership with the user-provided shared pointer.
    ///
    /// @param queue        The SYCL* queue object.
    /// @param data         The shared pointer to externally-allocated memory block.
    /// @param count        The number of elements of type :literal:`Data` in the memory block.
    /// @param dependencies Events that indicate when :literal:`Data` becomes ready to be read or written
    explicit array(const sycl::queue& queue,
                   const std::shared_ptr<T>& data,
                   std::int64_t count,
                   const std::vector<sycl::event>& dependencies = {})
            : impl_(new impl_t(detail::data_parallel_policy{ queue }, data, count)) {
        ONEDAL_ASSERT(sycl::get_pointer_type(data.get(), queue.get_context()) !=
                      sycl::usm::alloc::unknown);
        update_data(data.get(), count);
        sycl::event::wait_and_throw(dependencies);
    }
#endif

    /// Creates a new array instance that shares ownership with the user-provided shared pointer.
    ///
    /// @param data         The shared pointer to externally-allocated memory block.
    /// @param count        The number of elements of type :literal:`Data` in the memory block.
    explicit array(const std::shared_ptr<const T>& data, std::int64_t count)
            : impl_(new impl_t(detail::default_host_policy{}, data, count)) {
        update_data(data.get(), count);
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Creates a new array instance that shares ownership with the user-provided shared pointer.
    ///
    /// @param queue        The SYCL* queue object.
    /// @param data         The shared pointer to externally-allocated memory block.
    /// @param count        The number of elements of type :literal:`Data` in the memory block.
    /// @param dependencies Events that indicate when :literal:`Data` becomes ready to be read or
    ///                     written
    explicit array(const sycl::queue& queue,
                   const std::shared_ptr<const T>& data,
                   std::int64_t count,
                   const std::vector<sycl::event>& dependencies = {})
            : impl_(new impl_t(detail::data_parallel_policy{ queue }, data, count)) {
        ONEDAL_ASSERT(sycl::get_pointer_type(data.get(), queue.get_context()) !=
                      sycl::usm::alloc::unknown);
        update_data(data.get(), count);
        sycl::event::wait_and_throw(dependencies);
    }
#endif

    /// An aliasing constructor: creates a new array instance that stores :literal:`Data` pointer,
    /// assigns the pointer to the ownership structure of :literal:`ref` to the new instance. Array
    /// returns :literal:`Data` pointer as its mutable or immutable block depending on the
    /// :literal:`Data` type.
    ///
    /// @tparam Y    The type of elements in the referenced array.
    /// @tparam K    Either :literal:`T` or $const T$ type.
    ///
    /// @param ref   The array which shares ownership structure with created one.
    /// @param data  Mutable or immutable unmanaged pointer hold by created array.
    /// @param count The number of elements of type :literal:`T` in the :literal:`Data`.
    /// @pre  :expr:`std::is_same_v<data, const T> || std::is_same_v<data, T>`
    template <typename Y, typename K>
    explicit array(const array<Y>& ref, K* data, std::int64_t count)
            : impl_(new impl_t(*ref.impl_, data, count)) {
        update_data(impl_.get());
    }

    /// Replaces the :literal:`data`, :literal:`mutable_data` pointers, :literal:`count`, and
    /// pointer to the ownership structure in the array instance by the values in :literal:`other`.
    ///
    /// @post :expr:`data == other.data`
    /// @post :expr:`mutable_data == other.mutable_data`
    /// @post :expr:`count == other.count`
    array<T>& operator=(const array<T>& other) {
        array<T> tmp{ other };
        swap(*this, tmp);
        return *this;
    }

    /// Swaps the values of :literal:`data`, :literal:`mutable_data` pointers, :literal:`count`, and
    /// pointer to the ownership structure in the array instance and :literal:`other`.
    array<T>& operator=(array<T>&& other) {
        swap(*this, other);
        return *this;
    }

    /// The pointer to the memory block holding mutable data.
    /// @pre :expr:`has_mutable_data() == true`, othewise throws `domain_error`
    /// @invariant :expr:`mutable_data != nullptr` if :expr:`has_mutable_data() && count > 0`
    T* get_mutable_data() const {
        if (!has_mutable_data()) {
            throw domain_error(dal::detail::error_messages::array_does_not_contain_mutable_data());
        }
        return mutable_data_ptr_;
    }

    /// The pointer to the memory block holding immutable data.
    /// @invariant :expr:`data != nullptr` if :expr:`count > 0`
    /// @invariant if :expr:`has_mutable_data() == true` then :expr:`data == mutable_data`
    const T* get_data() const noexcept {
        return data_ptr_;
    }

    /// Returns whether array contains :literal:`mutable_data` or not
    ///
    /// @invariant :expr:`mutable_data != nullptr` if this returns `true` and :expr:`count > 0`
    bool has_mutable_data() const noexcept {
        return mutable_data_ptr_ != nullptr;
    }

    /// Returns mutable_data, if array contains it. Otherwise, allocates a
    /// memory block for mutable data and fills it with the data stored at :literal:`data`.
    /// Creates the ownership structure for allocated memory block and stores
    /// the pointer.
    ///
    /// @post :expr:`has_mutable_data() == true`
    array& need_mutable_data() {
        impl_->need_mutable_data();
        update_data(impl_->get_mutable_data(), impl_->get_count());
        return *this;
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Returns mutable_data, if array contains it. Otherwise, allocates a
    /// memory block for mutable data and fills it with the data stored at :literal:`data`.
    /// Creates the ownership structure for allocated memory block and stores
    /// the pointer.
    ///
    /// @param queue The SYCL* queue object.
    /// @param alloc The kind of USM to be allocated
    ///
    /// @post :expr:`has_mutable_data() == true`
    [[deprecated]] array& need_mutable_data(
        sycl::queue& queue,
        const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        impl_->need_mutable_data(detail::data_parallel_policy{ queue },
                                 detail::data_parallel_allocator<T>(queue, alloc));
        update_data(impl_->get_mutable_data(), impl_->get_count());
        return *this;
    }
#endif

    /// The number of elements of type :literal:`T` in a memory block
    std::int64_t get_count() const noexcept {
        return count_;
    }

    /// The size of memory block in bytes
    /// @invariant :expr:`size == count * sizeof(T)`
    std::int64_t get_size() const noexcept {
        return count_ * sizeof(T);
    }

    /// Resets ownership structure pointer to ``nullptr``,
    /// sets :literal:`count` to zero, :literal:`data` and :literal:`mutable_data` to :expr:`nullptr`.
    void reset() {
        impl_->reset();
        reset_data();
    }

    /// Allocates a new memory block for mutable data, does not initialize it,
    /// creates ownership structure for this block, assigns the structure inside the array.
    /// The array owns allocated memory block.
    ///
    /// @param count The number of elements of type :literal:`Data` to allocate memory for.
    void reset(std::int64_t count) {
        impl_->reset(detail::default_host_policy{}, count, detail::host_allocator<data_t>{});
        update_data(impl_->get_mutable_data(), count);
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Allocates a new memory block for mutable data, does not initialize it,
    /// creates ownership structure for this block, assigns the structure inside the array.
    /// The array owns allocated memory block.
    ///
    /// @param queue The SYCL* queue object.
    /// @param count The number of elements of type :literal:`T` to allocate memory for.
    /// @param alloc The kind of USM to be allocated
    void reset(const sycl::queue& queue,
               std::int64_t count,
               const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        impl_->reset(detail::data_parallel_policy{ queue },
                     count,
                     detail::data_parallel_allocator<T>(queue, alloc));
        update_data(impl_->get_mutable_data(), count);
    }
#endif

    /// Creates the ownership structure for memory block of externally-allocated mutable data,
    /// assigns input :literal:`deleter` object to it, sets :literal:`data` and
    /// :literal:`mutable_data` pointers to this block.
    ///
    /// @tparam Deleter     The type of a deleter used to free the :literal:`Data`.
    ///                     The deleter implements ``void operator()(Data*)`` member function.
    ///
    /// @param data         The mutable memory block pointer to be assigned inside the array
    /// @param count        The number of elements of type :literal:`Data` into the block
    /// @param deleter      The object used to free :literal:`Data`.
    template <typename Deleter>
    void reset(T* data, std::int64_t count, Deleter&& deleter) {
        impl_->reset(detail::default_host_policy{}, data, count, std::forward<Deleter>(deleter));
        update_data(data, count);
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Creates the ownership structure for memory block of externally-allocated mutable data,
    /// assigns input :literal:`deleter` object to it, sets :literal:`data` and
    /// :literal:`mutable_data` pointers to this block.
    ///
    /// @tparam Deleter     The type of a deleter used to free the :literal:`Data`.
    ///                     The deleter implements ``void operator()(Data*)`` member function.
    ///
    /// @param queue        The SYCL* queue object.
    /// @param data         The mutable memory block pointer to be assigned inside the array
    /// @param count        The number of elements of type :literal:`Data` into the block
    /// @param deleter      The object used to free :literal:`Data`.
    /// @param dependencies Events that indicate when :literal:`Data` becomes ready to be read or
    ///                     written
    template <typename Deleter>
    void reset(const sycl::queue& queue,
               T* data,
               std::int64_t count,
               Deleter&& deleter,
               const std::vector<sycl::event>& dependencies = {}) {
        ONEDAL_ASSERT(sycl::get_pointer_type(data, queue.get_context()) !=
                      sycl::usm::alloc::unknown);
        impl_->reset(detail::data_parallel_policy{ queue },
                     data,
                     count,
                     std::forward<Deleter>(deleter));
        update_data(data, count);
        sycl::event::wait_and_throw(dependencies);
    }
#endif

    /// Creates the ownership structure for memory block of externally-allocated immutable data,
    /// assigns input :literal:`deleter` object to it,
    /// sets :literal:`data` pointer to this block.
    ///
    /// @tparam ConstDeleter The type of a deleter used to free.
    ///                      The deleter implements `void operator()(const Data*)`` member function.
    ///
    /// @param data          The immutable memory block pointer to be assigned inside the array
    /// @param count         The number of elements of type :literal:`Data` into the block
    /// @param deleter       The object used to free :literal:`Data`.
    template <typename ConstDeleter>
    void reset(const T* data, std::int64_t count, ConstDeleter&& deleter) {
        impl_->reset(detail::default_host_policy{},
                     data,
                     count,
                     std::forward<ConstDeleter>(deleter));
        update_data(data, count);
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Creates the ownership structure for memory block of externally-allocated immutable data,
    /// assigns input :literal:`deleter` object to it,
    /// sets :literal:`data` pointer to this block.
    ///
    /// @tparam ConstDeleter The type of a deleter used to free.
    ///                      The deleter implements `void operator()(const Data*)`` member function.
    ///
    /// @param data          The immutable memory block pointer to be assigned inside the array
    /// @param count         The number of elements of type :literal:`Data` into the block
    /// @param deleter       The object used to free :literal:`Data`.
    /// @param dependencies  Events that indicate when :literal:`Data` becomes ready to be read or
    ///                      written
    template <typename ConstDeleter>
    void reset(const sycl::queue& queue,
               const T* data,
               std::int64_t count,
               ConstDeleter&& deleter,
               const std::vector<sycl::event>& dependencies = {}) {
        ONEDAL_ASSERT(sycl::get_pointer_type(data, queue.get_context()) !=
                      sycl::usm::alloc::unknown);
        impl_->reset(detail::data_parallel_policy{ queue },
                     data,
                     count,
                     std::forward<ConstDeleter>(deleter));
        update_data(data, count);
        sycl::event::wait_and_throw(dependencies);
    }
#endif

    /// Initializes :literal:`data` and :literal:`mutable_data` with data pointer,
    /// :literal:`count` with input :literal:`count` value, initializes
    /// the pointer to ownership structure with the one from ref. Array
    /// returns :literal:`Data` pointer as its mutable block.
    ///
    /// @tparam Y    The type of elements in the referenced array.
    ///
    /// @param ref   The array which is used to share ownership structure with current one.
    /// @param data  Mutable unmanaged pointer to be assigned to the array.
    /// @param count The number of elements of type :literal:`T` in the :literal:`Data`.
    template <typename Y>
    void reset(const array<Y>& ref, T* data, std::int64_t count) {
        impl_->reset(*ref.impl_, data, count);
        update_data(data, count);
    }

    /// Initializes :literal:`data` with data pointer,
    /// :literal:`count` with input :literal:`count` value, initializes
    /// the pointer to ownership structure with the one from ref. Array
    /// returns :literal:`Data` pointer as its immutable block.
    ///
    /// @tparam Y    The type of elements in the referenced array.
    ///
    /// @param ref   The array which is used to share ownership structure with current one.
    /// @param data  Immutable unmanaged pointer to be assigned to the array.
    /// @param count The number of elements of type :literal:`T` in the :literal:`Data`.
    template <typename Y>
    void reset(const array<Y>& ref, const T* data, std::int64_t count) {
        impl_->reset(*ref.impl_, data, count);
        update_data(data, count);
    }

    /// Provides a read-only access to the elements of array.
    /// Does not perform boundary checks.
    const T& operator[](std::int64_t index) const noexcept {
        ONEDAL_ASSERT(index < count_);
        ONEDAL_ASSERT(0 <= index);
        return data_ptr_[index];
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Returns a queue that was used to create array object.
    /// If no queue was provided at the array construction phase,
    /// returns empty :literal:`std::optional` object.
    std::optional<sycl::queue> get_queue() const {
        return impl_->get_queue();
    }
#endif

    /// Creates slice of this array
    array<T> get_slice(std::int64_t first, std::int64_t last) const {
        const auto slice_impl = impl_->get_slice(first, last);
        return array<T>{ new impl_t{ std::move(slice_impl) } };
    }

    /// @brief Creates array from impl
    array(impl_t* impl) : impl_(impl) {
        update_data(impl_.get());
    }

private:
    static void swap(array<T>& a, array<T>& b) {
        std::swap(a.impl_, b.impl_);
        std::swap(a.data_ptr_, b.data_ptr_);
        std::swap(a.mutable_data_ptr_, b.mutable_data_ptr_);
        std::swap(a.count_, b.count_);
    }

    void update_data(impl_t* impl) {
        if (impl->has_mutable_data()) {
            update_data(impl->get_mutable_data(), impl->get_count());
        }
        else {
            update_data(impl->get_data(), impl->get_count());
        }
    }

    void update_data(const T* data, std::int64_t count) noexcept {
        data_ptr_ = data;
        mutable_data_ptr_ = nullptr;
        count_ = count;
    }

    void update_data(T* data, std::int64_t count) noexcept {
        data_ptr_ = data;
        mutable_data_ptr_ = data;
        count_ = count;
    }

    void reset_data() noexcept {
        data_ptr_ = nullptr;
        mutable_data_ptr_ = nullptr;
        count_ = 0;
    }

    void serialize(detail::output_archive& ar) const {
        impl_->serialize(ar);
    }

    void deserialize(detail::input_archive& ar) {
        impl_->deserialize(ar);
        update_data(impl_.get());
    }

    detail::unique<impl_t> impl_;
    const T* data_ptr_;
    T* mutable_data_ptr_;
    std::int64_t count_;
};

template <typename Type>
struct is_array {
    constexpr static bool value = false;
};

template <typename Type>
struct is_array<array<Type>> {
    constexpr static bool value = true;
};

/// @tparam T Type to be checked for being an array
template <typename T>
constexpr bool is_array_v = is_array<T>::value;

/// @tparam T Type to be checked for being an array
template <typename T>
using enable_if_array_t = std::enable_if_t<is_array_v<T>>;

} // namespace v2

using v2::array;
using v2::is_array_v;
using v2::enable_if_array_t;

} // namespace oneapi::dal
