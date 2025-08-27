# Traditional DAAL Interface - AI Agents Context

> **Purpose**: This file provides context for AI agents working with the traditional DAAL interface, explaining the legacy patterns, memory management, and coding standards.

## 🏗️ DAAL Interface Architecture

The traditional DAAL interface provides a **CPU-focused, object-oriented** approach to machine learning algorithms with explicit memory management and SIMD optimizations.

### Key Characteristics
- **Memory Model**: Explicit memory management with smart pointers
- **Algorithm Pattern**: Training → Prediction → Model workflow
- **Data Structures**: Numeric tables and specialized containers
- **Threading**: Built-in threading layer abstraction

## 📁 Directory Structure

```
cpp/daal/
├── include/                 # Public interface headers
│   ├── algorithms/         # Algorithm implementations
│   │   ├── classifier/     # Classification algorithms
│   │   ├── clustering/     # Clustering algorithms
│   │   ├── regression/     # Regression algorithms
│   │   └── ...            # Other algorithm categories
│   ├── data_management/    # Data structures and management
│   │   ├── data/          # Core data containers
│   │   ├── features/      # Feature management
│   │   └── data_source/   # Data input/output
│   └── services/          # Utility services
│       ├── base.h         # Base service classes
│       ├── daal_defines.h # Common definitions
│       └── ...            # Other services
└── src/                    # Implementation files
    ├── algorithms/         # Algorithm implementations
    ├── data_management/    # Data management implementations
    └── services/          # Service implementations
```

## 🔧 Core Design Patterns

### 1. Algorithm Interface Pattern
All DAAL algorithms follow a consistent interface pattern:

```cpp
// Input/Output types
struct algorithm_input {
    // Input data and parameters
};

struct algorithm_result {
    // Results and computed values
};

// Main algorithm class
class algorithm_batch {
public:
    // Set input data
    void setInput(const algorithm_input& input);
    
    // Execute computation
    void compute();
    
    // Get results
    algorithm_result getResult();
    
    // Get algorithm parameters
    algorithm_parameter getParameter();
};
```

### 2. Memory Management Pattern
DAAL uses smart pointers for automatic memory management:

```cpp
#include "services/daal_shared_ptr.h"

// Use DAAL smart pointers
daal::services::SharedPtr<algorithm> algo;
daal::services::SharedPtr<numeric_table> data;

// Memory is automatically managed
algo = new algorithm();
data = new homogen_numeric_table<float>(nRows, nCols);
```

### 3. Data Management Pattern
Numeric tables are the primary data structure:

```cpp
#include "data_management/data/homogen_numeric_table.h"

// Create numeric table
auto table = new homogen_numeric_table<float>(nRows, nCols);

// Access data
float* data = table->getArray();
size_t nRows = table->getNumberOfRows();
size_t nCols = table->getNumberOfColumns();
```

## 📚 Algorithm Categories

### Classification
- **Decision Trees**: `decision_tree_classifier_training_batch`
- **Random Forest**: `random_forest_classifier_training_batch`
- **Support Vector Machine**: `svm_classifier_training_batch`
- **Naive Bayes**: `naive_bayes_classifier_training_batch`

### Clustering
- **K-Means**: `kmeans_batch`, `kmeans_distributed`
- **DBSCAN**: `dbscan_batch`
- **EM Algorithm**: `em_gmm_batch`

### Regression
- **Linear Regression**: `linear_regression_training_batch`
- **Ridge Regression**: `ridge_regression_training_batch`
- **Elastic Net**: `elastic_net_training_batch`

### Other Algorithms
- **Principal Component Analysis**: `pca_batch`
- **Covariance**: `covariance_batch`
- **Association Rules**: `apriori_batch`

## 🎯 Implementation Guidelines

### Header Files
- **Include Guards**: Use traditional `#ifndef` guards
- **Forward Declarations**: Minimize dependencies
- **Template Parameters**: Document all template parameters

### Source Files
- **Implementation**: Keep headers clean
- **Error Handling**: Use DAAL exception system
- **Memory Management**: Use DAAL smart pointers consistently

### Error Handling
```cpp
#include "services/error_handling.h"

// Check for errors
if (status != services::Status::OK) {
    // Handle error appropriately
    services::throwIfPossible(status);
}
```

## 🔍 Common Patterns and Best Practices

### 1. Algorithm Workflow
```cpp
// 1. Create algorithm instance
auto training = new algorithm_training_batch<float>();

// 2. Set input data
training->input.set(algorithm_training_input::data, data);
training->input.set(algorithm_training_input::labels, labels);

// 3. Execute training
training->compute();

// 4. Get results
auto result = training->getResult();
auto model = result->get(algorithm_training_result::model);
```

### 2. Data Preparation
```cpp
// Create numeric table from array
float* rawData = /* your data */;
auto table = new homogen_numeric_table<float>(rawData, nCols, nRows);

// Create data source for CSV files
auto dataSource = new csv_data_source<CSVFeatureManager>(filename);
auto table = dataSource->loadDataBlock();
```

### 3. Parameter Configuration
```cpp
// Set algorithm parameters
auto parameter = training->getParameter();
parameter->nClusters = 10;
parameter->maxIterations = 100;
parameter->accuracyThreshold = 1e-6;
```

## 🚫 Common Pitfalls to Avoid

### 1. Memory Management
- **Don't use** raw pointers for ownership
- **Don't forget** to check for null pointers
- **Do use** DAAL smart pointers consistently

### 2. Error Handling
- **Don't ignore** status codes
- **Don't assume** operations always succeed
- **Do check** status after every operation

### 3. Data Types
- **Don't mix** different numeric types
- **Don't assume** data layout compatibility
- **Do verify** data dimensions and types

## 🧪 Testing and Validation

### Unit Testing
- **Coverage**: Test all algorithm paths
- **Edge Cases**: Test boundary conditions
- **Error Conditions**: Test error handling

### Integration Testing
- **Examples**: Ensure examples build and run
- **Performance**: Validate performance characteristics
- **Memory**: Check for memory leaks

## 📖 Further Reading

- **[cpp/AGENTS.md](../AGENTS.md)** - Overall C++ implementation context
- **[cpp/oneapi/AGENTS.md](../oneapi/AGENTS.md)** - Modern oneAPI interface
- **[dev/AGENTS.md](../../dev/AGENTS.md)** - Build system and development tools
- **[docs/AGENTS.md](../../docs/AGENTS.md)** - Documentation guidelines

---

**Note**: This interface is maintained for backward compatibility. For new development, consider using the modern oneAPI interface.
