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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/primitives/ndshape.hpp"

namespace oneapi::dal::backend::primitives {

enum class ndorder {
    /// C-style ordering, row-major in 2D case
    c,
    /// Fortran-style ordering, column-major in 2D case
    f
};

template <ndorder order>
struct transposed_ndorder;

template <>
struct transposed_ndorder<ndorder::c> {
    static constexpr ndorder value = ndorder::f;
};

template <>
struct transposed_ndorder<ndorder::f> {
    static constexpr ndorder value = ndorder::c;
};

template <ndorder order>
constexpr ndorder transposed_ndorder_v = transposed_ndorder<order>::value;

template <std::int64_t axis_count, ndorder order = ndorder::c>
class ndarray_base : public base {
    static_assert(axis_count > 0, "Axis count must be non-zero");
    static_assert(order == ndorder::c || order == ndorder::f, "Only C or F orders are supported");

public:
    static constexpr std::int64_t axis_count_v = axis_count;

    ndarray_base() = default;

    explicit ndarray_base(const ndshape<axis_count>& shape, const ndshape<axis_count>& strides)
            : shape_(shape),
              strides_(strides) {
        if constexpr (order == ndorder::c) {
            ONEDAL_ASSERT(strides[axis_count - 1] == 1, "Last C-order stride must be 1");
        }
        else if constexpr (order == ndorder::f) {
            ONEDAL_ASSERT(strides[0] == 1, "First F-order stride must be 1");
        }

#ifdef ONEDAL_ENABLE_ASSERT
        const auto default_strides = get_default_strides(shape);
        for (std::int64_t i = 0; i < axis_count; i++) {
            ONEDAL_ASSERT(strides[i] >= default_strides[i],
                          "Custom stride must be greater than default");
        }
#endif
    }

    explicit ndarray_base(const ndshape<axis_count>& shape)
            : ndarray_base(shape, get_default_strides(shape)) {}

    constexpr ndorder get_order() const {
        return order;
    }

    const ndshape<axis_count>& get_shape() const {
        return shape_;
    }

    const ndshape<axis_count>& get_strides() const {
        return strides_;
    }

    std::int64_t get_dimension(std::int64_t axis) const {
        return shape_[axis];
    }

    std::int64_t get_stride(std::int64_t axis) const {
        return strides_[axis];
    }

    std::int64_t get_leading_stride() const {
        if constexpr (order == ndorder::c) {
            return strides_[0];
        }
        else if constexpr (order == ndorder::f) {
            return strides_[axis_count - 1];
        }
        return 0;
    }

    std::int64_t get_count() const {
        return get_shape().get_count();
    }

    template <typename... Indices>
    std::int64_t get_flat_index(Indices&&... indices) const {
        static_assert(std::tuple_size_v<std::tuple<Indices...>> == axis_count,
                      "Incorrect number of indices");

        std::int64_t flat_index = 0;
        ndindex<axis_count> idx{ std::int64_t(indices)... };

        for (std::int64_t i = 0; i < axis_count; i++) {
            ONEDAL_ASSERT(idx[i] < shape_[i], "Index is out of range");
            flat_index += strides_[i] * idx[i];
        }

        return flat_index;
    }

protected:
    static ndshape<axis_count> get_default_strides(const ndshape<axis_count>& shape) {
        // There is no need for multiplication overflow checks because
        // `shape` guaranties that shape[0] * ... * shape[axis_count - 1]
        // has no overflows

        ndindex<axis_count> strides;

        std::int64_t stride = 1;
        if constexpr (order == ndorder::c) {
            for (std::int64_t i = axis_count - 1; i >= 0; i--) {
                strides[i] = stride;
                stride *= shape[i];
            }
        }
        else if constexpr (order == ndorder::f) {
            for (std::int64_t i = 0; i < axis_count; i++) {
                strides[i] = stride;
                stride *= shape[i];
            }
        }

        return strides;
    }

    void check_if_strides_are_default() const {
#ifdef ONEDAL_ENABLE_ASSERT
        const auto default_strides = get_default_strides(this->get_shape());
        for (std::int64_t i = 0; i < axis_count; i++) {
            ONEDAL_ASSERT(this->get_strides()[i] == default_strides[i],
                          "Operation can be applied only to the array with default strides");
        }
#endif
    }

private:
    ndshape<axis_count> shape_;
    ndshape<axis_count> strides_;
};

template <typename T, std::int64_t axis_count, ndorder order = ndorder::c>
class ndarray;

/// The class represents a multi-dimensional view of an externally-defined memory block
/// with a fixed number of dimensions.
/// The view does not own the memory block and does not perform any memory management.
///
/// @tparam T           The type of elements in the view.
/// @tparam axis_count  The number of dimensions in the view.
/// @tparam order       C-contiguous (row-major) or FORTRAN-contiguous (column-major) order
///                     of the elements in 2-dimensional view.
template <typename T, std::int64_t axis_count, ndorder order = ndorder::c>
class ndview : public ndarray_base<axis_count, order> {
    static_assert(!std::is_const_v<T>, "T must be non-const");

    template <typename, std::int64_t, ndorder>
    friend class ndview;

    template <typename U>
    using enable_if_cv_match_t = std::enable_if_t<std::is_same_v<T, std::remove_cv_t<U>>>;

public:
    using base = ndarray_base<axis_count, order>;
    using shape_t = ndshape<axis_count>;

    ndview() : data_(nullptr) {}

    /// Creates a new multidimensional view instance by passing the pointer to externally-defined memory block
    /// for mutable data.
    ///
    /// @param data     The pointer to a homogeneous data block.
    /// @param shape    The shape of the created multidimensional array.
    ///
    /// @return The new multidimensional view instance.
    static ndview wrap(T* data, const shape_t& shape) {
        return ndview{ data, shape };
    }

    /// Creates a new multidimensional view instance by passing the pointer to externally-defined memory block
    /// for immutable data.
    ///
    /// @param data     The pointer to a homogeneous data block.
    /// @param shape    The shape of the created multidimensional array.
    ///
    /// @return The new multidimensional view instance.
    static ndview wrap(const T* data, const shape_t& shape) {
        return ndview{ data, shape };
    }

    /// Creates a new strided multidimensional view instance by passing the pointer to externally-defined
    /// memory block for mutable data.
    ///
    /// @param data     The pointer to a homogeneous data block.
    /// @param shape    The shape of the created multidimensional view.
    /// @param strides  The strides of the created multidimensional view.
    ///
    /// @return The new multidimensional view instance.
    static ndview wrap(T* data, const shape_t& shape, const shape_t& strides) {
        return ndview{ data, shape, strides };
    }

    /// Creates a new strided multidimensional view instance by passing the pointer to externally-defined
    /// memory block for immutable data.
    ///
    /// @param data     The pointer to a homogeneous data block.
    /// @param shape    The shape of the created multidimensional view.
    /// @param strides  The strides of the created multidimensional view.
    ///
    /// @return The new multidimensional view instance.
    static ndview wrap(const T* data, const shape_t& shape, const shape_t& strides) {
        return ndview{ data, shape, strides };
    }

    /// Creates a new multidimensional view instance from a mutable array.
    ///
    /// @param data    The array that stores a homogeneous data block.
    /// @param shape   The shape of the created multidimensional view.
    ///
    /// @return The new multidimensional view instance.
    static ndview wrap_mutable(const array<T>& data, const shape_t& shape) {
        ONEDAL_ASSERT(data.has_mutable_data());
        ONEDAL_ASSERT(data.get_count() >= shape.get_count());
        return wrap(data.get_mutable_data(), shape);
    }

    /// Creates a new strided multidimensional view instance from a mutable array.
    ///
    /// @param data    The array that stores a homogeneous data block.
    /// @param shape   The shape of the created multidimensional view.
    /// @param strides The strides of the created multidimensional view.
    ///
    /// @return The new multidimensional view instance.
    static ndview wrap_mutable(const array<T>& data, const shape_t& shape, const shape_t& strides) {
        ONEDAL_ASSERT(data.has_mutable_data());
        ONEDAL_ASSERT(data.get_count() >= shape.get_count());
        return wrap(data.get_mutable_data(), shape, strides);
    }

    /// Creates a new 1d view instance from an immutable array.
    ///
    /// @param data    The array that stores a homogeneous data block.
    ///
    /// @return The new 1d view instance.
    template <std::int64_t d = axis_count, typename = std::enable_if_t<d == 1>>
    static ndview wrap(const array<T>& data) {
        return wrap(data.get_data(), { data.get_count() });
    }

    /// Creates a new 1d view instance from a mutable array.
    ///
    /// @param data    The array that stores a homogeneous data block.
    ///
    /// @return The new 1d view instance.
    template <std::int64_t d = axis_count, typename = std::enable_if_t<d == 1>>
    static ndview wrap_mutable(const array<T>& data) {
        ONEDAL_ASSERT(data.has_mutable_data());
        return wrap(data.get_mutable_data(), { data.get_count() });
    }

    /// The pointer to the memory block holding immutable data.
    const T* get_data() const {
        return data_;
    }

    /// The pointer to the memory block holding mutable data.
    T* get_mutable_data() const {
        ONEDAL_ASSERT(data_is_mutable_);
        return const_cast<T*>(data_);
    }

    /// Returns whether the view contains data or not.
    bool has_data() const {
        return this->get_count() > 0;
    }

    /// Returns whether the view contains mutable data or not.
    bool has_mutable_data() const {
        return this->has_data() && this->data_is_mutable_;
    }

    /// Returns a reference to the element of 1d immutable view at specified location.
    ///
    /// @note This method does not perform boundary checks in release build.
    ///       In debug build, the method asserts that the index is not out of bounds.
    ///
    /// @note Should be used carefully in performance-critical parts of the code,
    ///       as inlining is not guaranteed.
    ///
    /// @param id  The index of the element to be returned.
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    const T& at(std::int64_t id) const {
        ONEDAL_ASSERT(has_data());
        ONEDAL_ASSERT((this->get_dimension(0) >= id) && (id >= 0));
        return *(get_data() + id);
    }

    /// Returns a reference to the element of 2d immutable view at specified location.
    ///
    /// @note This method does not perform boundary checks in release build.
    ///       In debug build, the method asserts that the index is not out of bounds.
    ///
    /// @note Should be used carefully in performance-critical parts of the code,
    ///       as inlining is not guaranteed.
    ///
    /// @param id0  The index of the element along the ``0``-th axis.
    /// @param id1  The index of the element along the ``1``-st axis.
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 2>>
    const T& at(std::int64_t id0, std::int64_t id1) const {
        ONEDAL_ASSERT(has_data());
        ONEDAL_ASSERT((this->get_dimension(0) >= id0) && (id0 >= 0));
        ONEDAL_ASSERT((this->get_dimension(1) >= id1) && (id1 >= 0));
        if constexpr (order == ndorder::c) {
            const auto* row = get_data() + id0 * this->get_stride(0);
            return *(row + id1);
        }
        else {
            const auto* const col = get_data() + id1 * this->get_stride(1);
            return *(col + id0);
        }
    }

    /// Returns a reference to the element of 1d mutable view at specified location.
    ///
    /// @param id  The index of the element to be returned.
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    T& at(std::int64_t id) {
        ONEDAL_ASSERT(has_mutable_data());
        ONEDAL_ASSERT((this->get_dimension(0) >= id) && (id >= 0));
        return *(get_mutable_data() + id);
    }

    /// Returns a reference to the element of 2d mutable view at specified location.
    ///
    /// @param id0  The index of the element along the ``0``-th axis.
    /// @param id1  The index of the element along the ``1``-st axis.
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 2>>
    T& at(std::int64_t id0, std::int64_t id1) {
        ONEDAL_ASSERT(has_mutable_data());
        ONEDAL_ASSERT((this->get_dimension(0) >= id0) && (id0 >= 0));
        ONEDAL_ASSERT((this->get_dimension(1) >= id1) && (id1 >= 0));
        if constexpr (order == ndorder::c) {
            auto* const row = get_mutable_data() + id0 * this->get_stride(0);
            return *(row + id1);
        }
        else {
            auto* const col = get_mutable_data() + id1 * this->get_stride(1);
            return *(col + id0);
        }
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Returns a copy of the element of 1d immutable view located in USM at specified location.
    /// The data is copied from the device to the host after the dependency events are completed.
    ///
    /// @param queue    The SYCL* queue object.
    /// @param id       The index of the element to be returned.
    /// @param deps     The vector of events that the operation depends on.
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    T at_device(sycl::queue& queue, std::int64_t id, const event_vector& deps = {}) const {
        return this->get_slice(id, id + 1).to_host(queue, deps).at(0);
    }

    /// Returns a copy of the element of 2d immutable view located in USM at specified location.
    /// The data is copied from the device to the host after the dependency events are completed.
    ///
    /// @param queue    The SYCL* queue object.
    /// @param id0      The index of the element along the ``0``-th axis.
    /// @param id1      The index of the element along the ``1``-st axis.
    /// @param deps     The vector of events that the operation depends on.
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 2>>
    T at_device(sycl::queue& queue,
                std::int64_t id0,
                std::int64_t id1,
                const event_vector& deps = {}) const {
        return this->get_row_slice(id0, id0 + 1)
            .get_col_slice(id1, id1 + 1)
            .to_host(queue, deps)
            .at(0, 0);
    }
#endif // ONEDAL_DATA_PARALLEL

    /// Get transposed multidimensional view.
    /// The shape and strides of the transposed multidimensional view are swapped.
    ///
    /// The data is not modified or copied: The transposed ndview points to the same memory block
    /// as the original ndview.
    ///
    /// @return The transposed multidimensional view.
    auto t() const {
        using tranposed_ndview_t = ndview<T, axis_count, transposed_ndorder_v<order>>;
        const auto& shape = this->get_shape();
        const auto& strides = this->get_strides();
        return tranposed_ndview_t{ data_, shape.t(), strides.t(), data_is_mutable_ };
    }

    /// Get the multidimensional view of the data reshaped to the requested shape.
    /// The total number of elements in the reshaped view must remain the same.
    /// The data is not copied: the reshaped ndview points to the same memory block
    /// as the original ndview.
    ///
    /// @tparam new_axis_count  The number of dimensions in the reshaped multidimensional view.
    ///
    /// @param new_shape        The shape of the reshaped multidimensional view.
    ///
    /// @return The reshaped multidimensional view.
    template <std::int64_t new_axis_count, ndorder new_order = order>
    auto reshape(const ndshape<new_axis_count>& new_shape) const {
        check_reshape_conditions(new_shape);
        using reshaped_ndview_t = ndview<T, new_axis_count, new_order>;
        return reshaped_ndview_t{ data_, new_shape, data_is_mutable_ };
    }

    /// Get 1-dimensional slice of 1-dimensional view.
    /// The slice is a view into the original memory block.
    /// The data is not copied: The sliced view points to the data within the same memory block
    /// as the original ndview.
    ///
    /// @param from The starting index of the data slice within the input view.
    /// @param to   The ending index of the data slice within the input view.
    ///
    /// @return The 1-dimensional view with a data slice.
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    ndview get_slice(std::int64_t from, std::int64_t to) const {
        ONEDAL_ASSERT((this->get_dimension(0) >= from) && (from >= 0));
        ONEDAL_ASSERT((this->get_dimension(0) >= to) && (to >= from));
        ONEDAL_ASSERT(this->has_data());
        const ndshape<1> new_shape{ to - from };
        const T* new_start_point = this->get_data() + from;
        return ndview(new_start_point, new_shape, this->get_strides(), this->data_is_mutable_);
    }

    /// Get 2-dimensional row slice of 2-dimensional view.
    /// The slice is a view into the original memory block.
    /// The data is not copied: The sliced view points to the data within the same memory block
    /// as the original ndview.
    ///
    /// @param from_row The starting row index of the data slice within the input view.
    /// @param to_row   The ending row index of the data slice within the input view.
    ///
    /// @return The 2-dimensional view with a data slice containing data rows [from_row, ..., to_row)
    ///         from the original view.
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 2>>
    ndview get_row_slice(std::int64_t from_row, std::int64_t to_row) const {
        ONEDAL_ASSERT((this->get_dimension(0) >= from_row) && (from_row >= 0));
        ONEDAL_ASSERT((this->get_dimension(0) >= to_row) && (to_row >= from_row));
        ONEDAL_ASSERT(this->has_data());
        const ndshape<2> new_shape{ (to_row - from_row), this->get_dimension(1) };
        if constexpr (order == ndorder::c) {
            const T* new_start_point = this->get_data() + from_row * this->get_leading_stride();
            return ndview(new_start_point, new_shape, this->get_strides(), this->data_is_mutable_);
        }
        const T* new_start_point = this->get_data() + from_row;
        return ndview(new_start_point, new_shape, this->get_strides(), this->data_is_mutable_);
    }

    /// Get 2-dimensional column slice of 2-dimensional view.
    /// The slice is a view into the original memory block.
    /// The data is not copied: The sliced view points to the data within the same memory block
    /// as the original ndview.
    ///
    /// @param from_col The starting column index of the data slice within the input view.
    /// @param to_col   The ending column index of the data slice within the input view.
    ///
    /// @return The 2-dimensional view with a data slice containing data columns [from_col, ..., to_col)
    ///         from the original view.
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 2>>
    ndview get_col_slice(std::int64_t from_col, std::int64_t to_col) const {
        return this->t().get_row_slice(from_col, to_col).t();
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Creates a multidimensional array with the same shape as the view,
    /// but having the data allocated on host. The data from the view is copied to the array on host.
    ///
    /// @param q    The SYCL* queue object.
    /// @param deps The vector of events that the operation depends on.
    ///
    /// @return The new multidimensional array instance.
    ndarray<T, axis_count, order> to_host(sycl::queue& q, const event_vector& deps = {}) const;

    /// Creates a multidimensional array with the same shape as the view,
    /// but having the data allocated in USM on device.
    /// The data from the view is copied to the array on device.
    ///
    /// @param q    The SYCL* queue object.
    /// @param deps The vector of events that the operation depends on.
    ///
    /// @return The new multidimensional array instance.
    ndarray<T, axis_count, order> to_device(sycl::queue& q, const event_vector& deps = {}) const;

    /// Prefetches the data from high bandwidth memory to local cache.
    /// Should be submitted ahead the expected computations to have enough time for data transfer.
    ///
    /// @param queue The SYCL* queue object.
    ///
    /// @return The SYCL* event object indicating the availability of the data in the view for reading
    ///         and writing.
    sycl::event prefetch(sycl::queue& queue) const {
        return queue.prefetch(data_, this->get_count());
    }
#endif

    /// Returns the pointer to the first element of the 1-dimensional view.
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    T* begin() {
        ONEDAL_ASSERT(data_is_mutable_);
        return get_mutable_data();
    }

    /// Returns the pointer to the past-the-last element of the 1-dimensional view.
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    T* end() {
        ONEDAL_ASSERT(data_is_mutable_);
        return get_mutable_data() + this->get_count();
    }

    /// Returns the constant pointer to the first element of the 1-dimensional view.
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    const T* cbegin() const {
        return get_data();
    }

    /// Returns the constant pointer to the past-the-last element of the 1-dimensional view.
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    const T* cend() const {
        return get_data() + this->get_count();
    }

protected:
    explicit ndview(const T* data,
                    const shape_t& shape,
                    const shape_t& strides,
                    bool data_is_mutable)
            : base(shape, strides),
              data_(data),
              data_is_mutable_(data_is_mutable) {}

    template <typename U, typename = enable_if_cv_match_t<U>>
    explicit ndview(U* data, const shape_t& shape, const shape_t& strides)
            : base(shape, strides),
              data_(data),
              data_is_mutable_(std::is_same_v<U, T>) {}

    explicit ndview(const T* data, const shape_t& shape, bool data_is_mutable)
            : base(shape),
              data_(data),
              data_is_mutable_(data_is_mutable) {}

    template <typename U, typename = enable_if_cv_match_t<U>>
    explicit ndview(U* data, const shape_t& shape)
            : base(shape),
              data_(data),
              data_is_mutable_(std::is_same_v<U, T>) {}

    template <std::int64_t new_axis_count>
    void check_reshape_conditions(const ndshape<new_axis_count>& new_shape) const {
        ONEDAL_ASSERT(new_shape.get_count() == this->get_count(),
                      "Total element count must remain unchanged");
    }

    ndview& set_mutability(bool data_is_mutable) {
        data_is_mutable_ = data_is_mutable;
        return *this;
    }

private:
    const T* data_;
    bool data_is_mutable_;
};

template <typename T1, ndorder ord1, typename T2, ndorder ord2>
inline void copy(ndview<T1, 2, ord1>& dst, const ndview<T2, 2, ord2>& src) {
    ONEDAL_ASSERT(src.has_data());
    ONEDAL_ASSERT(dst.has_mutable_data());
    const ndshape<2> dst_shape = dst.get_shape();
    ONEDAL_ASSERT(dst_shape == src.get_shape());
    if constexpr (ord1 == ndorder::c) {
        T1* const dst_ptr = dst.get_mutable_data();
        const T2* const src_ptr = src.get_data();
        const auto dst_stride = dst.get_leading_stride();
        const auto src_stride = src.get_leading_stride();

        for (std::int64_t r = 0; r < dst_shape[0]; ++r) {
            for (std::int64_t c = 0; c < dst_shape[1]; ++c) {
                T1& dst_ref = *(dst_ptr + r * dst_stride + c);
                if constexpr (ord2 == ndorder::c) {
                    dst_ref = static_cast<T1>(*(src_ptr + r * src_stride + c));
                }
                else {
                    dst_ref = static_cast<T2>(*(src_ptr + c * src_stride + r));
                }
            }
        }
    }
    else {
        auto new_dst = dst.t();
        const auto new_src = src.t();
        copy(new_dst, new_src);
    }
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename T1, ndorder ord1, typename T2, ndorder ord2>
inline sycl::event copy(sycl::queue& q,
                        ndview<T1, 2, ord1>& dst,
                        const ndview<T2, 2, ord2>& src,
                        const event_vector& deps = {}) {
    ONEDAL_ASSERT(src.has_data());
    ONEDAL_ASSERT(dst.has_mutable_data());
    const ndshape<2> dst_shape = dst.get_shape();
    ONEDAL_ASSERT(dst_shape == src.get_shape());
    sycl::event res_event;
    if constexpr (ord1 == ndorder::c) {
        T1* const dst_ptr = dst.get_mutable_data();
        const T2* const src_ptr = src.get_data();
        const auto dst_stride = dst.get_leading_stride();
        const auto src_stride = src.get_leading_stride();
        const auto cp_range = make_range_2d(dst_shape[0], dst_shape[1]);
        res_event = q.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for(cp_range, [=](sycl::id<2> idx) {
                T1& dst_ref = *(dst_ptr + idx[0] * dst_stride + idx[1]);
                if constexpr (ord2 == ndorder::c) {
                    dst_ref = static_cast<T1>(*(src_ptr + idx[0] * src_stride + idx[1]));
                }
                else {
                    dst_ref = static_cast<T2>(*(src_ptr + idx[1] * src_stride + idx[0]));
                }
            });
        });
    }
    else {
        auto new_dst = dst.t();
        const auto new_src = src.t();
        res_event = copy(q, new_dst, new_src, deps);
    }
    return res_event;
}

template <typename T1, ndorder ord1, typename T2, ndorder ord2>
inline sycl::event copy(sycl::queue& q,
                        ndview<T1, 1, ord1>& dst,
                        const ndview<T2, 1, ord2>& src,
                        const event_vector& deps = {}) {
    auto dst_2d = dst.template reshape<2>({ 1l, dst.get_count() });
    auto src_2d = src.template reshape<2>({ 1l, src.get_count() });

    return copy(q, dst_2d, src_2d, deps);
}

template <typename T>
inline sycl::event fill(sycl::queue& q,
                        ndview<T, 1>& dst,
                        const T& value = T{},
                        const event_vector& deps = {}) {
    ONEDAL_ASSERT(dst.has_mutable_data());
    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.fill(dst.get_mutable_data(), value, dst.get_count());
    });
}

template <typename T, ndorder ord1>
inline sycl::event fill(sycl::queue& q,
                        ndview<T, 2, ord1>& dst,
                        const T& value = T{},
                        const event_vector& deps = {}) {
    ONEDAL_ASSERT(dst.has_mutable_data());
    sycl::event res_event;
    if constexpr (ord1 == ndorder::c) {
        T* const dst_ptr = dst.get_mutable_data();
        const ndshape<2> dst_shape = dst.get_shape();
        const auto dst_stride = dst.get_leading_stride();
        const auto fl_range = make_range_2d(dst_shape[0], dst_shape[1]);
        res_event = q.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for(fl_range, [=](sycl::id<2> idx) {
                *(dst_ptr + idx[0] * dst_stride + idx[1]) = value;
            });
        });
    }
    else {
        auto new_dst = dst.t();
        res_event = fill(q, new_dst, value, deps);
    }
    return res_event;
}
#endif

/// Multidimensional array
///
/// @tparam T           The type of the memory block elements within the multidimensional array.
///
/// @tparam axis_count  The number of dimensions in the multidimensional array.
/// @tparam order       Row-major or column-major order of the 2-dimensional array.
template <typename T, std::int64_t axis_count, ndorder order>
class ndarray : public ndview<T, axis_count, order> {
    template <typename, std::int64_t, ndorder>
    friend class ndarray;

    using base = ndview<T, axis_count, order>;

    /// Type of the object holding the multidimensional array shape
    using shape_t = ndshape<axis_count>;

    /// Type of a shared pointer to :literal:`T`
    using shared_t = dal::detail::shared<T>;

    /// Type of the array with elements of type :literal:`T`
    using array_t = dal::array<std::remove_const_t<T>>;

    struct array_deleter {
        explicit array_deleter(const array_t& ary) : ary_(ary) {}

        explicit array_deleter(array_t&& ary) : ary_(std::move(ary)) {}

        void operator()(T* ptr) const {}

        array_t ary_;
    };

public:
    ndarray() = default;

    /// Creates a new multidimensional array instance by passing the pointer to externally-defined memory block
    /// for mutable data.
    ///
    /// @tparam Deleter     The type of a deleter called on ``data`` when
    ///                     the last ndarray that refers it is out of the scope.
    ///
    /// @param data         The pointer to a homogeneous data block.
    /// @param shape        The shape of the created multidimensional array.
    /// @param deleter      The deleter that is called on the ``data`` when the last ndarray that refers it
    ///                     is out of the scope.
    ///
    /// @return The new multidimensional array instance.
    template <typename Deleter = dal::detail::empty_delete<T>>
    static ndarray wrap(T* data, const shape_t& shape, Deleter&& deleter = Deleter{}) {
        auto shared = shared_t{ data, std::forward<Deleter>(deleter) };
        return wrap(std::move(shared), shape);
    }

    /// Creates a new multidimensional array instance by passing the pointer to externally-defined memory block
    /// for immutable data.
    ///
    /// @tparam Deleter     The type of a deleter called on ``data`` when
    ///                     the last ndarray that refers it is out of the scope.
    ///
    /// @param data         The pointer to a homogeneous data block.
    /// @param shape        The shape of the created multidimensional array.
    /// @param deleter      The deleter that is called on the ``data`` when the last ndarray that refers it
    ///                     is out of the scope.
    ///
    /// @return The new multidimensional array instance.
    template <typename Deleter = dal::detail::empty_delete<T>>
    static ndarray wrap(const T* data, const shape_t& shape, Deleter&& deleter = Deleter{}) {
        auto shared = shared_t{ const_cast<T*>(data), std::forward<Deleter>(deleter) };
        return wrap(std::move(shared), shape).set_mutability(false);
    }

    /// Creates a new multidimensional array instance by passing the shared pointer to externally-defined
    /// memory block for mutable data.
    ///
    /// @param data     The shared pointer to a homogeneous data block.
    /// @param shape    The shape of the created multidimensional array.
    ///
    /// @return The new multidimensional array instance.
    static ndarray wrap(const shared_t& data, const shape_t& shape) {
        return ndarray{ data, shape };
    }

    /// Creates a new multidimensional array instance by moving a shared pointer
    /// to externally-defined memory block for mutable data.
    ///
    /// @param data     R-value reference to the shared pointer to a homogeneous data block.
    /// @param shape    The shape of the created multidimensional array.
    ///
    /// @return The new multidimensional array instance.
    static ndarray wrap(shared_t&& data, const shape_t& shape) {
        return ndarray{ std::move(data), shape };
    }

    /// Creates a new multidimensional array instance from an immutable array.
    /// The created multidimensional array shares data ownership with the given array.
    ///
    /// @param ary      The array that stores a homogeneous data block.
    /// @param shape    The shape of the created multidimensional array.
    ///
    /// @return The new multidimensional array instance.
    static ndarray wrap(const array_t& ary, const shape_t& shape) {
        ONEDAL_ASSERT(ary.get_count() == shape.get_count());
        return wrap(ary.get_data(), shape, array_deleter{ ary });
    }

    /// Creates a new multidimensional array instance by moving an immutable input array.
    ///
    /// @param ary      The r-value reference to array that stores a homogeneous data block.
    /// @param shape    The shape of the created multidimensional array.
    ///
    /// @return The new multidimensional array instance.
    static ndarray wrap(array_t&& ary, const shape_t& shape) {
        ONEDAL_ASSERT(ary.get_count() == shape.get_count());
        const T* data_ptr = ary.get_data();
        return wrap(data_ptr, shape, array_deleter{ std::move(ary) });
    }

    /// Creates a 1d ndarray instance from an immutable array.
    /// The created ndarray shares data ownership with the given array.
    ///
    /// @param ary  The array that stores a homogeneous data block.
    ///
    /// @return The new 1d ndarray instance.
    static ndarray wrap(const array_t& ary) {
        static_assert(axis_count == 1);
        return wrap(ary, shape_t{ ary.get_count() });
    }

    /// Creates a 1d ndarray instance by moving an immutable input array.
    /// The created ndarray shares data ownership with the given array.
    ///
    /// @param ary  The r-value reference to array that stores a homogeneous data block.
    ///
    /// @return The new 1d ndarray instance.
    static ndarray wrap(array_t&& ary) {
        static_assert(axis_count == 1);
        std::int64_t ary_count = ary.get_count();
        return wrap(std::move(ary), shape_t{ ary_count });
    }

    /// Creates a new multidimensional array instance from a mutable array.
    /// The created multidimensional array shares data ownership with the given array.
    ///
    /// @param ary      The array that stores a homogeneous data block.
    /// @param shape    The shape of the created multidimensional array.
    ///
    /// @return The new multidimensional array instance.
    static ndarray wrap_mutable(const array_t& ary, const shape_t& shape) {
        ONEDAL_ASSERT(ary.get_count() == shape.get_count());
        auto shared = shared_t{ ary.get_mutable_data(), array_deleter{ ary } };
        return wrap(std::move(shared), shape);
    }

    /// Creates a new multidimensional array instance by moving a mutable input array.
    ///
    /// @param ary      The r-value reference to array that stores a homogeneous data block.
    /// @param shape    The shape of the created multidimensional array.
    ///
    /// @return The new multidimensional array instance.
    static ndarray wrap_mutable(array_t&& ary, const shape_t& shape) {
        ONEDAL_ASSERT(ary.get_count() == shape.get_count());
        T* data_ptr = ary.get_mutable_data();
        auto shared = shared_t{ data_ptr, array_deleter{ std::move(ary) } };
        return wrap(std::move(shared), shape);
    }

    /// Creates a 1d ndarray instance from a mutable array.
    /// The created ndarray shares data ownership with the given array.
    ///
    /// @param ary  The array that stores a homogeneous data block.
    ///
    /// @return The new 1d ndarray instance.
    static ndarray wrap_mutable(const array_t& ary) {
        static_assert(axis_count == 1);
        return wrap_mutable(ary, shape_t{ ary.get_count() });
    }

    /// Creates a 1d ndarray instance by moving a mutable input array.
    /// The created ndarray shares data ownership with the given array.
    ///
    /// @param ary  The array that stores a homogeneous data block.
    ///
    /// @return The new 1d ndarray instance.
    static ndarray wrap_mutable(array_t&& ary) {
        static_assert(axis_count == 1);
        std::int64_t ary_count = ary.get_count();
        return wrap_mutable(std::move(ary), shape_t{ ary_count });
    }

    /// Creates an uninitialized multidimensional array of a requested shape.
    ///
    /// @param shape The shape of the created multidimensional array.
    ///
    /// @return The new multidimensional array instance.
    static ndarray empty(const shape_t& shape) {
        T* ptr = dal::detail::malloc<T>(dal::detail::default_host_policy{}, shape.get_count());
        return wrap(ptr,
                    shape,
                    dal::detail::make_default_delete<T>(dal::detail::default_host_policy{}));
    }

    /// Creates a multidimensional array of the requested shape and copies the values
    /// from the input pointer to a memory block into that multidimensional array.
    ///
    /// @param data     The pointer to a homogeneous data block.
    /// @param shape    The shape of the created multidimensional array.
    ///
    /// @return The new multidimensional array instance.
    static ndarray copy(const T* data, const shape_t& shape) {
        auto ary = empty(shape);
        ary.assign(data, shape.get_count());
        return ary;
    }

    /// Creates a multidimensional array of the requested shape and fills it with zeros.
    ///
    /// @param shape    The shape of the created multidimensional array.
    ///
    /// @return The new multidimensional array instance.
    static ndarray zeros(const shape_t& shape) {
        auto ary = empty(shape);
        ary.fill(T(0));
        return ary;
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Creates an uninitialized multidimensional array of a requested shape
    /// with the elements stored in SYCL* USM.
    ///
    /// @param q            The SYCL* queue object.
    /// @param shape        The shape of the created multidimensional array.
    /// @param alloc_kind   The kind of USM to be allocated.
    ///
    /// @return The new multidimensional array instance.
    static ndarray empty(const sycl::queue& q,
                         const shape_t& shape,
                         const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        T* ptr = malloc<T>(q, shape.get_count(), alloc_kind);
        return wrap(ptr, shape, usm_deleter<T>{ q });
    }

    /// Creates a multidimensional array of the requested shape with the elements
    /// stored in SYCL* USM and copies the values from the input pointer to a memory block
    /// into that multidimensional array.
    ///
    /// @param q            The SYCL* queue object.
    /// @param data         The pointer to a homogeneous data block.
    /// @param shape        The shape of the created multidimensional array.
    /// @param alloc_kind   The kind of USM to be allocated.
    ///
    /// @return A tuple with the created multidimensional array and the SYCL* event object
    ///         indicating the availability of the resulting array for reading and writing.
    static std::tuple<ndarray, sycl::event> copy(
        sycl::queue& q,
        const T* data,
        const shape_t& shape,
        const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        auto ary = empty(q, shape, alloc_kind);
        auto event = ary.assign(q, data, shape.get_count());
        return { ary, event };
    }

    /// Creates a multidimensional array of the requested shape with the elements
    /// stored in SYCL* USM and fills it with a scalar value provided.
    ///
    /// @param q            The SYCL* queue object.
    /// @param shape        The shape of the created multidimensional array.
    /// @param value        The scalar value to fill the array with.
    /// @param alloc_kind   The kind of USM to be allocated.
    /// @param deps         The vector of events that the operation depends on.
    ///
    /// @return A tuple with the created multidimensional array and the SYCL* event object
    ///         indicating the availability of the resulting array for reading and writing.
    static std::tuple<ndarray, sycl::event> full(
        sycl::queue& q,
        const shape_t& shape,
        const T& value,
        const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared,
        const event_vector& deps = {}) {
        auto ary = empty(q, shape, alloc_kind);
        auto event = ary.fill(q, value, deps);
        return { ary, event };
    }

    /// Creates a multidimensional array of the requested shape with the elements
    /// stored in SYCL* USM and fills it with zeros.
    ///
    /// @param q            The SYCL* queue object.
    /// @param shape        The shape of the created multidimensional array.
    /// @param alloc_kind   The kind of USM to be allocated.
    ///
    /// @return A tuple with the created multidimensional array and the SYCL* event object
    ///         indicating the availability of the resulting array for reading and writing.
    static std::tuple<ndarray, sycl::event> zeros(
        sycl::queue& q,
        const shape_t& shape,
        const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        return full(q, shape, T(0), alloc_kind);
    }

    /// Creates a multidimensional array of the requested shape with the elements
    /// stored in SYCL* USM and fills it with ones.
    ///
    /// @param q            The SYCL* queue object.
    /// @param shape        The shape of the created multidimensional array.
    /// @param alloc_kind   The kind of USM to be allocated.
    ///
    /// @return A tuple with the created multidimensional array and the SYCL* event object
    ///         indicating the availability of the resulting array for reading and writing.
    static std::tuple<ndarray, sycl::event> ones(
        sycl::queue& q,
        const shape_t& shape,
        const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        return full(q, shape, T(1), alloc_kind);
    }
#endif

    /// Get ndview sub-object of the ndarray.
    ///
    /// @return The ndview sub-object.
    const base& get_view() const {
        return *this;
    }

    /// Get 1d dal::array that shares the ownership on the data block of this ndarray.
    ///
    /// @return The 1d dal::array that contains all the data from this multidimentional array.
    array_t flatten() const {
        return array_t{ data_, this->get_count() };
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Get 1d dal::array that shares the ownership on the SYCL* USM data block of this ndarray.
    ///
    /// @param q    The SYCL* queue object.
    /// @param deps The vector of events that the operation depends on.
    ///
    /// @return The 1d dal::array that contains all the USM data from this multidimentional array.
    array_t flatten(sycl::queue& q, const event_vector& deps = {}) const {
        ONEDAL_ASSERT(is_known_usm(q, data_.get()));
        return array_t{ q, data_, this->get_count(), deps };
    }
#endif

    /// Get transposed multidimensional array.
    /// The shape and strides of the transposed multidimensional array are swapped.
    ///
    /// The data is not copied: The transposed ndarray shares the ownership on the data block
    /// with the original ndarray.
    ///
    /// @return The transposed multidimensional array.
    auto t() const {
        using tranposed_ndarray_t = ndarray<T, axis_count, transposed_ndorder_v<order>>;
        const auto& shape = this->get_shape();
        const auto& strides = this->get_strides();
        return tranposed_ndarray_t{ data_, shape.t(), strides.t() }.set_mutability(
            this->has_mutable_data());
    }

    /// Get the multidimensional array reshaped to the requested shape.
    /// The total number of elements in the reshaped array must remain the same.
    /// The data is not copied: the reshaped ndarray shares the ownership on the data block
    /// with the original ndarray.
    ///
    /// @tparam new_axis_count  The number of dimensions in the reshaped multidimensional array.
    ///
    /// @param new_shape        The shape of the reshaped multidimensional array.
    ///
    /// @return The reshaped multidimensional array.
    template <std::int64_t new_axis_count>
    auto reshape(const ndshape<new_axis_count>& new_shape) const {
        using reshaped_ndarray_t = ndarray<T, new_axis_count, order>;
        base::check_reshape_conditions(new_shape);
        return reshaped_ndarray_t{ data_, new_shape }.set_mutability(this->has_mutable_data());
    }

    /// Fill multidimensional array with a scalar value.
    ///
    /// @param value The scalar value to fill the array with.
    void fill(T value) {
        T* data_ptr = this->get_mutable_data();
        for (std::int64_t i = 0; i < this->get_count(); i++) {
            data_ptr[i] = value;
        }
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Fill multidimensional array with a scalar value.
    ///
    /// @param q     The SYCL* queue object.
    /// @param value The scalar value to fill the array with.
    /// @param deps  The vector of events that the operation depends on.
    ///
    /// @return The SYCL* event object indicating the availability of the array
    ///         for reading and writing.
    sycl::event fill(sycl::queue& q, T value, const event_vector& deps = {}) {
        auto data_ptr = this->get_mutable_data();
        ONEDAL_ASSERT(is_known_usm(q, data_ptr));
        return q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.fill(data_ptr, value, this->get_count());
        });
    }
#endif

    /// Fill multidimensional array with the values from a sequence 0, 1, 2, ..., N-1,
    /// where N is the total number of elements in the array.
    void arange() {
        T* data_ptr = this->get_mutable_data();
        for (std::int64_t i = 0; i < this->get_count(); i++) {
            data_ptr[i] = i;
        }
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Fill multidimensional array with the values from a sequence 0, 1, 2, ..., N-1,
    /// where N is the total number of elements in the array.
    ///
    /// @param q     The SYCL* queue object.
    /// @param deps  The vector of events that the operation depends on.
    ///
    /// @return The SYCL* event object indicating the availability of the array
    ///         for reading and writing.
    sycl::event arange(sycl::queue& q, const event_vector& deps = {}) {
        auto data_ptr = this->get_mutable_data();
        ONEDAL_ASSERT(is_known_usm(q, data_ptr));
        return q.submit([&](sycl::handler& cgh) {
            const auto range = dal::backend::make_range_1d(this->get_count());
            cgh.depends_on(deps);
            cgh.parallel_for(range, [=](sycl::id<1> idx) {
                data_ptr[idx] = idx;
            });
        });
    }
#endif

    /// Copy the values from the input pointer to a memory block into multidimensional array.
    ///
    /// @param source_ptr    The pointer to a homogeneous data block.
    /// @param source_count  The number of elements in the input memory block.
    void assign(const T* source_ptr, std::int64_t source_count) {
        ONEDAL_ASSERT(source_ptr != nullptr);
        ONEDAL_ASSERT(source_count > 0);
        ONEDAL_ASSERT(source_count <= this->get_count());
        return dal::backend::copy(this->get_mutable_data(), source_ptr, source_count);
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Copy the values from the input pointer to SYCL* USM memory block into multidimensional array.
    ///
    /// @param q             The SYCL* queue object.
    /// @param source_ptr    The pointer to a homogeneous data block.
    /// @param source_count  The number of elements in the input memory block.
    /// @param deps          The vector of events that the operation depends on.
    ///
    /// @return The SYCL* event object indicating the availability of the array
    ///         for reading and writing.
    sycl::event assign(sycl::queue& q,
                       const T* source_ptr,
                       std::int64_t source_count,
                       const event_vector& deps = {}) {
        ONEDAL_ASSERT(source_ptr != nullptr);
        ONEDAL_ASSERT(source_count > 0);
        ONEDAL_ASSERT(source_count <= this->get_count());
        return dal::backend::copy(q, this->get_mutable_data(), source_ptr, source_count, deps);
    }

    /// Copy the values from the input pointer to a memory block allocated on host
    /// into multidimensional array.
    ///
    /// @param q             The SYCL* queue object.
    /// @param source_ptr    The pointer to a homogeneous data block allocated in host memory.
    /// @param source_count  The number of elements in the input memory block.
    /// @param deps          The vector of events that the operation depends on.
    ///
    /// @return The SYCL* event object indicating the availability of the array
    ///         for reading and writing.
    sycl::event assign_from_host(sycl::queue& q,
                                 const T* source_ptr,
                                 std::int64_t source_count,
                                 const event_vector& deps = {}) {
        ONEDAL_ASSERT(source_ptr != nullptr);
        ONEDAL_ASSERT(source_count > 0);
        ONEDAL_ASSERT(source_count <= this->get_count());
        return dal::backend::copy_host2usm(q,
                                           this->get_mutable_data(),
                                           source_ptr,
                                           source_count,
                                           deps);
    }

    /// Copy the values from the input multidimensional array
    /// into this multidimensional array.
    ///
    /// @param q    The SYCL* queue object.
    /// @param src  The multidimensional array to copy data from.
    /// @param deps The vector of events that the operation depends on.
    ///
    /// @return The SYCL* event object indicating the availability of the array
    ///         for reading and writing.
    sycl::event assign(sycl::queue& q, const ndarray& src, const event_vector& deps = {}) {
        ONEDAL_ASSERT(src.get_count() > 0);
        ONEDAL_ASSERT(src.get_count() <= this->get_count());
        return this->assign(q, src.get_data(), src.get_count(), deps);
    }

    /// Copy the values from the input multidimensional array containing data allocated on host
    /// into this multidimensional array.
    ///
    /// @param q    The SYCL* queue object.
    /// @param src  The multidimensional array to copy data from.
    /// @param deps The vector of events that the operation depends on.
    ///
    /// @return The SYCL* event object indicating the availability of the array
    ///         for reading and writing.
    sycl::event assign_from_host(sycl::queue& q,
                                 const ndarray& src,
                                 const event_vector& deps = {}) {
        ONEDAL_ASSERT(src.get_count() > 0);
        ONEDAL_ASSERT(src.get_count() <= this->get_count());
        return this->assign_from_host(q, src.get_data(), src.get_count(), deps);
    }
#endif

    /// Get a slice of the multidimensional array along the specified axis.
    /// The slice is a view into the original multidimensional array.
    /// The data is not copied: The sliced ndarray shares the ownership on the data block.
    ///
    /// @param offset The offset along the specified axis.
    /// @param count  The number of elements along the specified axis.
    /// @param axis   The axis to slice along. Only axis ``0`` is supported.
    ///
    /// @return The multidimensional array with a data slice.
    ndarray slice(std::int64_t offset, std::int64_t count, std::int64_t axis = 0) const {
        ONEDAL_ASSERT(order == ndorder::c, "Only C-order is supported");
        ONEDAL_ASSERT(axis == 0, "Non-zero axis is not supported");
        ONEDAL_ASSERT(offset >= 0);
        ONEDAL_ASSERT(count > 0);
        ONEDAL_ASSERT(offset + count <= this->get_dimension(axis));

        const auto shape = this->get_shape();
        const std::int64_t rest_shape_count = shape.get_count() / shape[axis];
        ONEDAL_ASSERT(rest_shape_count > 0);

        T* data_ptr = data_.get() + offset * rest_shape_count;
        const auto aliased_data = shared_t{ data_, data_ptr };

        backend::ndindex<axis_count> shape_index = shape.get_index();
        shape_index[0] = count;

        return wrap(aliased_data, ndshape<axis_count>{ shape_index });
    }

#ifdef ONEDAL_DATA_PARALLEL

    /// Split a multidimensional array into multiple slices along the specified axis.
    /// The slices are views into the original multidimensional array.
    /// The data is not copied: The sliced ndarrays share the ownership on the data block.
    ///
    /// @param fold_count The number of slices to split the multidimensional array into.
    /// @param axis       The axis to split along. Only axis ``0`` is supported.
    ///
    /// @return A vector of multidimensional arrays with data slices.
    std::vector<ndarray> split(std::int64_t fold_count, std::int64_t axis = 0) const {
        ONEDAL_ASSERT(order == ndorder::c, "Only C-order is supported");
        ONEDAL_ASSERT(axis == 0, "Non-zero axis is not supported");
        ONEDAL_ASSERT(fold_count >= 0);

        if (fold_count <= 0) {
            return {};
        }

        const std::int64_t regular_block = this->get_dimension(axis) / fold_count;
        ONEDAL_ASSERT(regular_block > 0);

        std::vector<ndarray> slices;
        slices.reserve(fold_count);

        for (std::int64_t i = 0; i < fold_count - 1; i++) {
            slices.push_back(this->slice(i * regular_block, regular_block, axis));
        }

        {
            const std::int64_t i = fold_count - 1;
            const std::int64_t tail_block = this->get_dimension(axis) - regular_block * i;
            slices.push_back(this->slice(i * regular_block, tail_block, axis));
        }

        return slices;
    }
#endif

private:
    explicit ndarray(const shared_t& data, const shape_t& shape, const shape_t& strides)
            : base(data.get(), shape, strides),
              data_(data) {}

    explicit ndarray(shared_t&& data, const shape_t& shape, const shape_t& strides)
            : base(data.get(), shape, strides),
              data_(std::move(data)) {}

    explicit ndarray(const shared_t& data, const shape_t& shape)
            : base(data.get(), shape),
              data_(data) {}

    explicit ndarray(shared_t&& data, const shape_t& shape)
            : base(data.get(), shape),
              data_(std::move(data)) {}

    ndarray& set_mutability(bool data_is_mutable) {
        base::set_mutability(data_is_mutable);
        return *this;
    }

    shared_t data_;
};

#ifdef ONEDAL_DATA_PARALLEL

template <typename T, std::int64_t axis_count, ndorder order>
ndarray<T, axis_count, order> ndview<T, axis_count, order>::to_host(
    sycl::queue& q,
    const event_vector& deps) const {
    T* host_ptr = dal::detail::host_allocator<T>().allocate(this->get_count());
    dal::backend::copy_usm2host(q, host_ptr, this->get_data(), this->get_count(), deps)
        .wait_and_throw();
    return ndarray<T, axis_count, order>::wrap(
        host_ptr,
        this->get_shape(),
        dal::detail::make_default_delete<T>(dal::detail::default_host_policy{}));
}

template <typename T, std::int64_t axis_count, ndorder order>
ndarray<T, axis_count, order> ndview<T, axis_count, order>::to_device(
    sycl::queue& q,
    const event_vector& deps) const {
    auto dev = ndarray<T, axis_count, order>::empty(q, this->get_shape(), sycl::usm::alloc::device);
    dal::backend::copy_host2usm(q,
                                dev.get_mutable_data(),
                                this->get_data(),
                                this->get_count(),
                                deps)
        .wait_and_throw();
    return dev;
}

template <ndorder yorder,
          typename Type,
          ndorder xorder,
          sycl::usm::alloc alloc = sycl::usm::alloc::device>
inline auto copy(sycl::queue& q,
                 const ndview<Type, 2, xorder>& src,
                 const event_vector& deps = {}) {
    ONEDAL_ASSERT(src.has_data());
    const auto shape = src.get_shape();
    auto res_array = ndarray<Type, 2, yorder>::empty(q, shape, alloc);
    auto res_event = copy(q, res_array, src, deps);
    return std::make_pair(res_array, res_event);
}

template <typename Type, ndorder xorder, sycl::usm::alloc alloc = sycl::usm::alloc::device>
inline auto copy(sycl::queue& q,
                 const ndview<Type, 2, xorder>& src,
                 const event_vector& deps = {}) {
    return copy<xorder, Type, xorder, alloc>(q, src, deps);
}

#endif

} // namespace oneapi::dal::backend::primitives

namespace oneapi::dal::backend {

template <typename Type>
inline Type* begin(const primitives::ndview<Type, 1>& arr) {
    ONEDAL_ASSERT(arr.has_mutable_data());
    return arr.get_mutable_data();
}

template <typename Type>
inline Type* end(const primitives::ndview<Type, 1>& arr) {
    return begin(arr) + arr.get_count();
}

template <typename Type>
inline const Type* cbegin(const primitives::ndview<Type, 1>& arr) {
    ONEDAL_ASSERT(arr.has_data());
    return arr.get_data();
}

template <typename Type>
inline const Type* cend(const primitives::ndview<Type, 1>& arr) {
    return cbegin(arr) + arr.get_count();
}

} // namespace oneapi::dal::backend
