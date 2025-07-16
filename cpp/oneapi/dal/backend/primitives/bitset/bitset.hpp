#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::backend::primitives {

/**
 * @brief A bitset class that provides a set of operations on a bitset.
 */
template <typename ElementType = std::uint32_t>
class bitset {
public:
    using element_t = ElementType;
    static constexpr size_t element_bitsize = sizeof(element_t) * 8; // Number of bits in an element

    bitset(element_t* data) : _data(data) {}

    /// Sets the bit at the specified index to 1.
    inline void set(std::uint32_t index) const {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        _data[element_index] |= (element_t(1) << bit_index);
    }

    /// Clears the bit at the specified index to 0.
    inline void clear(std::uint32_t index) const {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        _data[element_index] &= ~(element_t(1) << bit_index);
    }

    /// Checks if the bit at the specified index is set.
    inline bool test(std::uint32_t index) const {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        return (_data[element_index] & (element_t(1) << bit_index)) != 0;
    }

    inline bool gas(std::uint32_t index) const {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        bool set = (_data[element_index] & (element_t(1) << bit_index)) != 0;
        if (!set) {
            _data[element_index] &= ~(element_t(1) << bit_index); // Clear
        }
        return set;
    }

    /// Sets the bit at the specified index using atomic operations.
    template <sycl::memory_order mem_order = sycl::memory_order::relaxed,
              sycl::memory_scope mem_scope = sycl::memory_scope::device>
    inline void atomic_set(std::uint32_t index) const {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        sycl::atomic_ref<element_t, mem_order, mem_scope, sycl::access::address_space::local_space>
            atomic_element(_data[element_index]);
        atomic_element.fetch_or(element_t(1) << bit_index);
    }

    /// Clears the bit at the specified index using atomic operations.
    template <sycl::memory_order mem_order = sycl::memory_order::relaxed,
              sycl::memory_scope mem_scope = sycl::memory_scope::device>
    inline void atomic_clear(std::uint32_t index) {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        sycl::atomic_ref<element_t, mem_order, mem_scope, sycl::access::address_space::local_space>
            atomic_element(_data[element_index]);
        atomic_element.fetch_and(~(element_t(1) << bit_index));
    }

    /// Checks if the bit at the specified index is set using atomic operations.
    template <sycl::memory_order mem_order = sycl::memory_order::relaxed,
              sycl::memory_scope mem_scope = sycl::memory_scope::device>
    inline bool atomic_test(std::uint32_t index) const {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        sycl::atomic_ref<element_t, mem_order, mem_scope, sycl::access::address_space::local_space>
            atomic_element(_data[element_index]);
        return (atomic_element.load() & (element_t(1) << bit_index)) != 0;
    }

    template <sycl::memory_order mem_order = sycl::memory_order::relaxed,
              sycl::memory_scope mem_scope = sycl::memory_scope::device>
    inline bool atomic_gas(std::uint32_t index) const {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        sycl::atomic_ref<element_t, mem_order, mem_scope, sycl::access::address_space::local_space>
            atomic_element(_data[element_index]);
        bool set = (atomic_element.load() & (element_t(1) << bit_index)) != 0;
        if (!set) {
            atomic_element.fetch_and(~(element_t(1) << bit_index)); // Clear
        }
        return set;
    }

    /// Returns the pointer to the underlying data.
    inline element_t* get_data() {
        return _data;
    }

    inline void set_data(element_t* data) {
        _data = data;
    }

    /// Returns the pointer to the underlying data (const version).
    inline const element_t* get_data() const {
        return _data;
    }

    /// override operator []
    inline element_t& operator[](std::size_t index) {
        return _data[index];
    }

private:
    element_t* _data;
};

} // namespace oneapi::dal::backend::primitives