# Modern oneAPI Interface - AI Agents Context

> **Purpose**: Context for AI agents working with the modern oneAPI interface, SYCL integration, and modern C++ practices.

## 🏗️ oneAPI Interface Architecture

The modern oneAPI interface provides **GPU-accelerated, modern C++** machine learning with heterogeneous computing support.

### Key Characteristics
- **Headers**: `.hpp` files with `#pragma once` guards
- **Memory**: STL smart pointers (`std::unique_ptr`, `std::shared_ptr`) with RAII
- **Error Handling**: C++ exceptions extending STL (`std::invalid_argument`, `std::domain_error`)
- **GPU Support**: Intel SYCL for heterogeneous CPU/GPU computing
- **Distributed**: MPI integration via `oneapi::dal::preview::spmd`
- **Versioning**: `namespace v1` for stable API, `preview` for experimental features

### Heterogeneous Computing
```cpp
// Multi-backend execution with SYCL
namespace oneapi::dal {
    // CPU execution
    template <typename... Args>
    auto train(Args&&... args) {
        return dal::detail::train_dispatch(std::forward<Args>(args)...);
    }
    
    // GPU execution via SYCL queue
    template <typename... Args>
    auto train(sycl::queue& queue, Args&&... args) {
        return dal::backend::dispatch<dal::backend::gpu>(
            queue, std::forward<Args>(args)...);
    }
    
    // Distributed execution
    template <typename... Args>
    auto train(const spmd::communicator<>& comm, Args&&... args) {
        return dal::spmd::train_dispatch(comm, std::forward<Args>(args)...);
    }
}
```

## 🎭 Advanced oneAPI Template Patterns

### 1. Type-Safe Template Dispatching
```cpp
// Modern template metaprogramming with perfect forwarding
template <typename Context, typename Float, typename Method, typename Task>
struct train_ops_dispatcher {
    train_result<Task> operator()(const Context& ctx,
                                  const descriptor_base<Task>& desc,
                                  const train_parameters<Task>& params,
                                  const train_input<Task>& input) const {
        // Compile-time dispatch based on context
        if constexpr (std::is_same_v<Context, dal::detail::cpu_context>) {
            return cpu_kernel<Float, Method>::compute(desc, params, input);
        } else if constexpr (std::is_same_v<Context, dal::detail::gpu_context>) {
            return gpu_kernel<Float, Method>::compute(ctx.get_queue(), desc, params, input);
        }
    }
};

// Algorithm operations with SFINAE
template <typename Descriptor>
struct train_ops {
    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const 
        -> std::enable_if_t<dal::detail::is_valid_context_v<Context>, result_t> {
        return train_ops_dispatcher<Context, float_t, method_t, task_t>{}(ctx, desc, input);
    }
};
```

### 2. SYCL Integration with USM
```cpp
// SYCL kernel with Unified Shared Memory
template <typename Float>
class sycl_kmeans_kernel {
public:
    sycl::event operator()(sycl::queue& queue, 
                          const sycl::usm::shared_ptr<Float>& data,
                          const sycl::usm::shared_ptr<Float>& centroids,
                          std::int64_t n_samples, std::int64_t n_features) {
        
        return queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(n_samples, 256), 
                           [=](sycl::nd_item<1> item) {
                const auto idx = item.get_global_id(0);
                if (idx < n_samples) {
                    // GPU-optimized distance computation
                    Float min_distance = std::numeric_limits<Float>::max();
                    std::int32_t closest_centroid = 0;
                    // ... computation logic
                    assignments[idx] = closest_centroid;
                }
            });
        });
    }
};
```

### 3. Modern Exception Hierarchy
```cpp
// Custom exception types extending STL
namespace oneapi::dal {
    class invalid_argument : public std::invalid_argument {
    public:
        explicit invalid_argument(const std::string& message) 
            : std::invalid_argument(message) {}
    };
    
    class unimplemented : public std::logic_error {
    public:
        explicit unimplemented(const std::string& message) 
            : std::logic_error(message) {}
    };
    
    // Error message factory
    namespace detail::error_messages {
        inline std::string input_data_is_empty() {
            return "Input data table is empty";
        }
    }
}
```

## 🏛️ oneAPI Design Patterns

### 1. Descriptor Pattern with Fluent Interface
```cpp
// Builder pattern with method chaining
template <typename Float, typename Method = method::lloyd_dense, typename Task = task::clustering>
class descriptor : public descriptor_base<Task> {
public:
    // Fluent interface for configuration
    descriptor& set_cluster_count(std::int64_t value) {
        if (value <= 0) {
            throw invalid_argument("Cluster count must be positive");
        }
        cluster_count_ = value;
        return *this;
    }
    
    descriptor& set_max_iteration_count(std::int64_t value) {
        max_iteration_count_ = value;
        return *this;
    }
    
    // Type-safe getters
    std::int64_t get_cluster_count() const noexcept { return cluster_count_; }
    
private:
    std::int64_t cluster_count_ = 2;
    std::int64_t max_iteration_count_ = 100;
};
```

### 2. Table Abstraction with Type Safety
```cpp
// Modern data table with RAII
template <typename Data = std::byte>
class homogen_table {
public:
    // Factory methods with perfect forwarding
    template <typename T>
    static homogen_table wrap(const T* data, std::int64_t row_count, std::int64_t column_count) {
        static_assert(std::is_arithmetic_v<T>, "Data type must be arithmetic");
        auto array_data = array<T>::wrap(data, row_count * column_count);
        return homogen_table(std::move(array_data), row_count, column_count);
    }
    
    // Move semantics
    homogen_table(homogen_table&&) = default;
    homogen_table& operator=(homogen_table&&) = default;
};

// Type-safe accessor pattern
template <typename T>
class row_accessor {
public:
    explicit row_accessor(const table& source_table);
    
    // Range-based access with automatic memory management
    array<T> pull(const range& rows = { 0, -1 }) const;
    void push(const array<T>& data, const range& rows = { 0, -1 });
};
```

### 3. Distributed Computing with MPI
```cpp
// Modern distributed computing abstraction
namespace oneapi::dal::preview::spmd {
    template <device_memory_access MemoryAccess = device_memory_access::usm>
    class communicator {
    public:
        explicit communicator(MPI_Comm mpi_comm);
        
        std::int64_t get_rank() const noexcept;
        std::int64_t get_rank_count() const noexcept;
        
        // Type-safe collective operations
        template <typename T>
        void allreduce(const T* send_buf, T* recv_buf, std::int64_t count) const;
        
        // SYCL-aware distributed operations
        template <typename T>
        sycl::event allreduce(sycl::queue& queue,
                             const sycl::usm::shared_ptr<T>& send_buf,
                             sycl::usm::shared_ptr<T>& recv_buf,
                             std::int64_t count) const;
    };
    
    inline communicator<> make_communicator(MPI_Comm comm = MPI_COMM_WORLD);
}
```

## ⚡ oneAPI Performance Features

### 1. Unified Shared Memory (USM)
```cpp
// Modern memory management for CPU/GPU interoperability
template <typename T>
class usm_array {
    sycl::usm::shared_ptr<T> data_;
    sycl::queue queue_;
    
public:
    explicit usm_array(sycl::queue& q, std::int64_t n) : queue_(q) {
        data_ = sycl::usm::allocate_shared<T>(queue_, n);
    }
    
    // CPU/GPU accessible pointer
    T* get_mutable_data() const { return data_.get(); }
    
    // Asynchronous memory operations
    sycl::event copy_from_host(const T* host_data);
};
```

### 2. Backend Abstraction
```cpp
// Runtime backend selection
namespace oneapi::dal::backend {
    template <typename Backend>
    struct dispatch {
        template <typename... Args>
        static auto compute(Args&&... args) {
            if constexpr (std::is_same_v<Backend, cpu>) {
                return cpu_impl::compute(std::forward<Args>(args)...);
            } else if constexpr (std::is_same_v<Backend, gpu>) {
                return gpu_impl::compute(std::forward<Args>(args)...);
            }
        }
    };
    
    // Automatic backend selection based on data location
    template <typename... Args>
    auto smart_dispatch(Args&&... args) {
        if (detail::has_gpu_data(args...)) {
            return dispatch<gpu>::compute(std::forward<Args>(args)...);
        } else {
            return dispatch<cpu>::compute(std::forward<Args>(args)...);
        }
    }
}
```

## 🔧 Core oneAPI Interface

### Modern Algorithm Pattern
```cpp
#include "oneapi/dal/algo/kmeans.hpp"

// Fluent descriptor configuration
auto desc = kmeans::descriptor<float>()
    .set_cluster_count(10)
    .set_max_iteration_count(100)
    .set_accuracy_threshold(1e-6);

// Synchronous CPU training
auto train_result = train(desc, train_data);

// Asynchronous GPU training
sycl::queue gpu_queue(sycl::gpu_selector_v);
auto gpu_result = train(gpu_queue, desc, gpu_data);

// Distributed training
auto comm = spmd::make_communicator();
auto dist_result = train(comm, desc, local_data);
```

### Data Table Pattern
```cpp
#include "oneapi/dal/table/homogen.hpp"

// Create table from data with type safety
auto table = homogen_table::wrap(data, row_count, column_count);

// Safe data access
auto accessor = row_accessor<const float>(table);
auto row_data = accessor.pull({0, 10}); // Rows 0-9
```

## 🎯 Critical oneAPI Rules

- **Memory**: STL RAII with `std::unique_ptr`/`std::shared_ptr`, never raw pointers
- **Error Handling**: C++ exceptions extending STL hierarchy
- **Headers**: `.hpp` with `#pragma once`, `oneapi::dal` namespaces
- **GPU**: SYCL integration with USM for zero-copy CPU/GPU operations
- **Distributed**: MPI integration via `spmd::communicator` for multi-node computing
- **Type Safety**: Template metaprogramming with compile-time dispatch

## 📖 Further Reading
- **[AGENTS.md](/AGENTS.md)** - Repository overview and context
- **[cpp/AGENTS.md](/cpp/AGENTS.md)** - C++ implementation overview  
- **[cpp/daal/AGENTS.md](/cpp/daal/AGENTS.md)** - Traditional DAAL interface
- **[dev/AGENTS.md](/dev/AGENTS.md)** - Build system architecture
- **[dev/bazel/AGENTS.md](/dev/bazel/AGENTS.md)** - Bazel-specific patterns
- **[docs/AGENTS.md](/docs/AGENTS.md)** - Documentation generation
- **[examples/AGENTS.md](/examples/AGENTS.md)** - Example integration patterns