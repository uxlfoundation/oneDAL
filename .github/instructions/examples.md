# Examples Instructions for GitHub Copilot

## Example Code Guidelines

### Code Quality Standards
- **Readability**: Clear, well-commented code
- **Completeness**: Self-contained, runnable examples
- **Best Practices**: Follow oneDAL coding guidelines
- **C++ Standard**: Use C++17 maximum (no C++20/23 features for compatibility)
- **Error Handling**: Proper error checking and handling
- **Documentation**: Clear header comments and inline explanations

### Example Structure
- **Header Comments**: Clear description of example purpose
- **Includes**: All necessary headers and dependencies
- **Main Function**: Well-structured main function
- **Error Handling**: Proper error checking and validation
- **Output**: Clear output and results demonstration

## Interface-Specific Examples

### oneAPI Interface Examples (Recommended)

#### Basic Algorithm Example
```cpp
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/io/csv.hpp"
#include <iostream>

using namespace oneapi::dal;

int main() {
    try {
        auto train_data = read<homogen_table>("train_data.csv");
        auto desc = kmeans::descriptor<float>()
            .set_cluster_count(10)
            .set_max_iteration_count(100);
        
        auto train_result = train(desc, train_data);
        auto test_data = read<homogen_table>("test_data.csv");
        auto infer_result = infer(desc, train_result.get_model(), test_data);
        
        std::cout << "Clusters: " << train_result.get_cluster_count() << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

#### SYCL GPU Example
```cpp
#include <sycl/sycl.hpp>
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include <iostream>

using namespace oneapi::dal;

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        auto data = read<homogen_table>("data.csv");
        auto desc = kmeans::descriptor<float>().set_cluster_count(10);
        
        auto result = train(q, desc, data);
        std::cout << "GPU clusters: " << result.get_cluster_count() << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

#### Distributed Computing Example
```cpp
#include <mpi.h>
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/spmd/communicator.hpp"
#include <iostream>

using namespace oneapi::dal;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    try {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        auto comm = spmd::make_communicator();
        auto local_data = read<homogen_table>("local_data.csv");
        auto desc = kmeans::descriptor<float>().set_cluster_count(10);
        auto result = train(comm, desc, local_data);
        
        if (rank == 0) {
            std::cout << "Distributed clusters: " << result.get_cluster_count() << std::endl;
        }
        
        MPI_Finalize();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Process " << rank << " error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }
}
```

### DAAL Interface Examples (Legacy)

#### Basic Algorithm Example
```cpp
#include "algorithms/kmeans/kmeans_batch.h"
#include "data_management/data/homogen_numeric_table.h"
#include <iostream>

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

int main() {
    try {
        auto data = new homogen_numeric_table<float>(10, 1000);
        auto kmeans = new kmeans::Batch<float>();
        
        auto parameter = kmeans->getParameter();
        parameter->nClusters = 10;
        
        kmeans->input.set(kmeans::BatchInput::data, data);
        kmeans->compute();
        
        auto result = kmeans->getResult();
        std::cout << "Clusters: " << parameter->nClusters << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

## Data Management Examples

### Data Loading
```cpp
// CSV Data Loading (oneAPI)
auto data = read<homogen_table>("data.csv");

// CSV Data Loading (DAAL)
auto dataSource = new CSVDataSource<CSVFeatureManager>("data.csv");
auto table = dataSource->loadDataBlock();

// Array Data Creation (oneAPI)
float raw_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
auto table = homogen_table::wrap(raw_data, 2, 3); // 2 rows, 3 columns

// Array Data Creation (DAAL)
float raw_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
auto table = new homogen_numeric_table<float>(raw_data, 3, 2); // 3 columns, 2 rows
```

### Data Validation
```cpp
// Validate data dimensions
if (data.get_row_count() == 0 || data.get_column_count() == 0) {
    throw std::invalid_argument("Data table is empty");
}

// Validate data types
if (data.get_metadata().get_data_type(0) != data_type::float32) {
    throw std::invalid_argument("Data must be float32 type");
}

// Check for NaN values
auto accessor = row_accessor<const float>(data);
for (std::int64_t i = 0; i < data.get_row_count(); ++i) {
    auto row = accessor.pull({i, i + 1});
    for (auto val : row) {
        if (std::isnan(val)) {
            throw std::invalid_argument("Data contains NaN values");
        }
    }
}
```

## Performance Examples

### CPU Optimization
```cpp
// Enable SIMD optimizations
auto desc = kmeans::descriptor<float>()
    .set_cluster_count(10)
    .set_max_iteration_count(100)
    .set_accuracy_threshold(1e-6);

// Use appropriate data layout
auto table = homogen_table::wrap(data, rows, cols); // Row-major layout for better cache performance
```

### Memory Management
```cpp
// Efficient memory allocation
auto data = std::make_unique<float[]>(size);
auto table = homogen_table::wrap(data.get(), rows, cols);

// Use move semantics
auto table1 = homogen_table::wrap(data1, rows, cols);
auto table2 = std::move(table1); // Efficient transfer of ownership
```

## Testing and Validation Examples

### Unit Test Example
```cpp
#include <gtest/gtest.h>
#include "oneapi/dal/algo/kmeans.hpp"

class KMeansTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test data
        float test_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        data_ = homogen_table::wrap(test_data, 2, 3);
    }
    
    homogen_table data_;
};

TEST_F(KMeansTest, BasicClustering) {
    auto desc = kmeans::descriptor<float>()
        .set_cluster_count(2);
    
    auto result = train(desc, data_);
    
    EXPECT_EQ(result.get_cluster_count(), 2);
    EXPECT_GT(result.get_iteration_count(), 0);
}
```

### Integration Test Example
```cpp
#include <gtest/gtest.h>
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/io/csv.hpp"

TEST(IntegrationTest, CSVToKMeans) {
    // Load data from CSV
    auto data = read<homogen_table>("test_data.csv");
    
    // Perform clustering
    auto desc = kmeans::descriptor<float>().set_cluster_count(3);
    auto result = train(desc, data);
    
    // Validate results
    EXPECT_EQ(result.get_cluster_count(), 3);
    EXPECT_GT(data.get_row_count(), 0);
    EXPECT_GT(data.get_column_count(), 0);
}
```

## Common Patterns

### Error Handling Pattern
```cpp
try {
    // Main algorithm execution
    auto result = train(desc, data);
    
    // Validate results
    if (result.get_cluster_count() != expected_clusters) {
        throw std::runtime_error("Unexpected number of clusters");
    }
    
    return result;
} catch (const std::exception& e) {
    std::cerr << "Algorithm failed: " << e.what() << std::endl;
    throw;
}
```

### Resource Management Pattern
```cpp
class AlgorithmRunner {
private:
    std::unique_ptr<algorithm_descriptor> desc_;
    std::unique_ptr<data_table> data_;
    
public:
    AlgorithmRunner(std::unique_ptr<algorithm_descriptor> desc,
                   std::unique_ptr<data_table> data)
        : desc_(std::move(desc))
        , data_(std::move(data)) {}
    
    auto run() {
        return train(*desc_, *data_);
    }
    
    // Destructor automatically cleans up resources
};
```

## Best Practices

1. **Complete Examples**: Always provide complete, runnable examples
2. **Error Handling**: Include proper error checking and validation
3. **Documentation**: Add clear comments explaining complex logic
4. **Testing**: Include unit tests for example functionality
5. **Performance**: Consider performance implications of example code
6. **Platform Independence**: Avoid platform-specific code when possible

## Common Pitfalls

1. **Incomplete Code**: Don't provide code snippets that can't run
2. **Missing Dependencies**: Don't forget to include necessary headers
3. **Poor Error Handling**: Don't ignore potential error conditions
4. **Memory Leaks**: Don't forget proper resource cleanup
5. **Platform Dependencies**: Don't hardcode platform-specific paths
6. **C++20/23 Features**: Don't use C++20/23 features - stick to C++17 maximum for compatibility

---

**Note**: Examples are crucial for user adoption and understanding. Always ensure examples are complete, correct, and follow best practices.
