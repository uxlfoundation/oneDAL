
# Traditional DAAL Interface - AI Agents Context

> **Purpose**: Context for AI agents working with the traditional DAAL interface, legacy patterns, and memory management.

## 🏗️ DAAL Interface Architecture

The traditional DAAL interface provides **CPU-focused, object-oriented** approach to machine learning with explicit memory management and SIMD optimizations.

### Key Characteristics
- **Memory Model**: Explicit memory management with smart pointers
- **Algorithm Pattern**: Training → Prediction → Model workflow
- **Data Structures**: Numeric tables and specialized containers
- **Threading**: Built-in threading layer abstraction

## 📁 Structure
```
cpp/daal/
├── include/            # Public interface headers
│   ├── algorithms/     # Algorithm implementations
│   ├── data_management/ # Data structures and management
│   └── services/       # Utility services
└── src/                # Implementation files
```

## 🔧 Core Design Patterns

### Algorithm Interface Pattern
All DAAL algorithms follow a consistent interface:

```cpp
// Main algorithm class
class algorithm_batch {
public:
    void setInput(const algorithm_input& input);  // Set input data
    void compute();                               // Execute computation
    algorithm_result getResult();                 // Get results
    algorithm_parameter getParameter();           // Get parameters
};
```

### Memory Management Pattern
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

### Data Management Pattern
Numeric tables are the primary data structure:

```cpp
#include "data_management/data/homogen_numeric_table.h"

// Create and access numeric table
auto table = new homogen_numeric_table<float>(nRows, nCols);
float* data = table->getArray();
size_t nRows = table->getNumberOfRows();
```

## 📚 Algorithm Categories

### Classification
- **Decision Trees**: `decision_tree_classifier_training_batch`
- **Random Forest**: `random_forest_classifier_training_batch`
- **Support Vector Machine**: `svm_classifier_training_batch`
- **Naive Bayes**: `naive_bayes_classifier_training_batch`

### Clustering & Regression
- **K-Means**: `kmeans_batch`, `kmeans_distributed`
- **DBSCAN**: `dbscan_batch`
- **Linear Regression**: `linear_regression_training_batch`
- **Ridge Regression**: `ridge_regression_training_batch`

### Other Algorithms
- **PCA**: `pca_batch`
- **Covariance**: `covariance_batch`
- **Association Rules**: `apriori_batch`

## 🎯 Implementation Guidelines

### Header Files
- **Include Guards**: Use traditional `#ifndef` guards
- **Forward Declarations**: Minimize dependencies
- **Template Parameters**: Document all template parameters

### Source Files & Error Handling
- **Implementation**: Keep headers clean
- **Error Handling**: Use DAAL exception system
- **Memory Management**: Use DAAL smart pointers consistently

```cpp
#include "services/error_handling.h"

// Check for errors
if (status != services::Status::OK) {
    services::throwIfPossible(status);
}
```

## 🔍 Common Patterns

### Algorithm Workflow
```cpp
// 1. Create algorithm instance
auto training = new algorithm_training_batch<float>();

// 2. Set input data
training->input.set(algorithm_training_input::data, data);
training->input.set(algorithm_training_input::labels, labels);

// 3. Execute training and get results
training->compute();
auto result = training->getResult();
auto model = result->get(algorithm_training_result::model);
```

### Data Preparation
```cpp
// Create numeric table from array
float* rawData = /* your data */;
auto table = new homogen_numeric_table<float>(rawData, nCols, nRows);

// Create data source for CSV files
auto dataSource = new csv_data_source<CSVFeatureManager>(filename);
auto table = dataSource->loadDataBlock();
```

### Parameter Configuration
```cpp
// Set algorithm parameters
auto parameter = training->getParameter();
parameter->nClusters = 10;
parameter->maxIterations = 100;
parameter->accuracyThreshold = 1e-6;
```

## 🚫 Common Pitfalls
- **Memory Management**: Use DAAL smart pointers, don't use raw pointers for ownership
- **Error Handling**: Don't ignore status codes, check status after every operation
- **Data Types**: Don't mix different numeric types, verify data dimensions and types

## 🧪 Testing and Validation
- **Unit Testing**: Test all algorithm paths, edge cases, error conditions
- **Integration Testing**: Ensure examples build and run, validate performance, check for memory leaks

## 📖 Further Reading
- **[cpp/AGENTS.md](../AGENTS.md)** - Overall C++ implementation context
- **[cpp/oneapi/AGENTS.md](../oneapi/AGENTS.md)** - Modern oneAPI interface
- **[dev/AGENTS.md](../../dev/AGENTS.md)** - Build system and development tools

---

**Note**: This interface is maintained for backward compatibility. For new development, consider using the modern oneAPI interface.
