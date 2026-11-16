/* file: zip_iterator.h */
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
//  Implementation of zip iterator and related utilities.
//  Can be replaced with std::zip_view::iterator in C++23.
//--
*/

#pragma once

#include <functional> // std::less
#include <iterator>
#include <tuple>
#include <utility>

/// Proxy reference to the same position in several parallel arrays. Assignment and swap act on
/// the pointed-to elements, so ZipIterator sorts all the arrays in place without copies.
template <typename... T>
struct ZipRef
{
    using Seq = std::index_sequence_for<T...>;

    std::tuple<T *...> ptr;

    explicit ZipRef(const std::tuple<T *...> & p) : ptr(p) {}
    ZipRef(const ZipRef &) = default;

    operator std::tuple<T...>() const { return load(Seq()); }

    ZipRef & operator=(const std::tuple<T...> & v)
    {
        store(v, Seq());
        return *this;
    }

    // Assigns the elements, not the pointers.
    ZipRef & operator=(const ZipRef & other)
    {
        assign(other, Seq());
        return *this;
    }

    // Found by ADL from std::iter_swap, which is called on proxy prvalues.
    friend void swap(ZipRef a, ZipRef b) { a.swapWith(b, Seq()); }

private:
    template <std::size_t... K>
    std::tuple<T...> load(std::index_sequence<K...>) const
    {
        return std::tuple<T...>(*std::get<K>(ptr)...);
    }
    template <std::size_t... K>
    void store(const std::tuple<T...> & v, std::index_sequence<K...>)
    {
        ((*std::get<K>(ptr) = std::get<K>(v)), ...);
    }
    template <std::size_t... K>
    void assign(const ZipRef & other, std::index_sequence<K...>)
    {
        ((*std::get<K>(ptr) = *std::get<K>(other.ptr)), ...);
    }
    template <std::size_t... K>
    void swapWith(ZipRef & other, std::index_sequence<K...>)
    {
        (std::swap(*std::get<K>(ptr), *std::get<K>(other.ptr)), ...);
    }
};

// TODO: Replace with std::zip_view::iterator (C++23).
template <typename... T>
class ZipIterator
{
    using Seq = std::index_sequence_for<T...>;

public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = std::tuple<T...>;
    using difference_type   = std::ptrdiff_t;
    using reference         = ZipRef<T...>;
    using pointer           = void;

    ZipIterator() : _ptr() {}
    explicit ZipIterator(T *... ptr) : _ptr(ptr...) {}

    reference operator*() const { return reference(_ptr); }
    reference operator[](difference_type d) const { return *(*this + d); }

    ZipIterator & operator+=(difference_type d)
    {
        advance(d, Seq());
        return *this;
    }
    ZipIterator & operator-=(difference_type d) { return *this += -d; }
    ZipIterator & operator++() { return *this += 1; }
    ZipIterator & operator--() { return *this -= 1; }
    ZipIterator operator++(int)
    {
        ZipIterator tmp(*this);
        ++*this;
        return tmp;
    }
    ZipIterator operator--(int)
    {
        ZipIterator tmp(*this);
        --*this;
        return tmp;
    }

    friend ZipIterator operator+(ZipIterator it, difference_type d) { return it += d; }
    friend ZipIterator operator+(difference_type d, ZipIterator it) { return it += d; }
    friend ZipIterator operator-(ZipIterator it, difference_type d) { return it -= d; }
    friend difference_type operator-(const ZipIterator & a, const ZipIterator & b) { return a.head() - b.head(); }

    friend bool operator==(const ZipIterator & a, const ZipIterator & b) { return a.head() == b.head(); }
    friend bool operator!=(const ZipIterator & a, const ZipIterator & b) { return a.head() != b.head(); }
    friend bool operator<(const ZipIterator & a, const ZipIterator & b) { return a.head() < b.head(); }
    friend bool operator>(const ZipIterator & a, const ZipIterator & b) { return a.head() > b.head(); }
    friend bool operator<=(const ZipIterator & a, const ZipIterator & b) { return a.head() <= b.head(); }
    friend bool operator>=(const ZipIterator & a, const ZipIterator & b) { return a.head() >= b.head(); }

private:
    template <std::size_t... K>
    void advance(difference_type d, std::index_sequence<K...>)
    {
        ((std::get<K>(_ptr) += d), ...);
    }
    auto head() const { return std::get<0>(_ptr); }

    std::tuple<T *...> _ptr;
};

// The sorts compare proxies and values in every combination; only the first array (the key) is read.
struct ZipKeyLess
{
    template <typename... T>
    static auto keyOf(const ZipRef<T...> & r)
    {
        return *std::get<0>(r.ptr);
    }
    template <typename... T>
    static auto keyOf(const std::tuple<T...> & v)
    {
        return std::get<0>(v);
    }

    template <typename A, typename B>
    bool operator()(const A & a, const B & b) const
    {
        return keyOf(a) < keyOf(b);
    }
};
