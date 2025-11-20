# Deprecation of DAAL API layers of certain algorithms in oneDAL 2025.10

## Introduction

While certain classes are present in the DAAL API, these classes are intended
for internal use and provide limited value to end users.
Maintaining these APIs in the public interface reduces product flexibility
and clarity.

In contrast to [20251015-DAAL-algorithms-deprecation](https://github.com/uxlfoundation/oneDAL/tree/rfcs/rfcs/20251015-DAAL-algorithms-deprecation),
this RFC proposes first deprecating these API layers, then removing them
from DAAL API by moving to the ``daal::*::internal`` namespace while preserving
all underlying algorithmic computational kernels, as these kernels are used
internally by other algorithms.

## Proposal

The following APIs will be marked as deprecated in the next minor oneDAL 2025.10,
with complete API removal scheduled for the next major release, oneDAL 2026.0:

#### [Distributions](https://uxlfoundation.github.io/oneDAL/daal/algorithms/distributions/index.html)

- [distributions::Input](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distributions/distribution_types.h#L88)
- [distributions::Result](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distributions/distribution_types.h#L128)

- [distributions::bernoulli::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distributions/bernoulli/bernoulli.h#L89)
- [distributions::bernoulli::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distributions/bernoulli/bernoulli.h#L56)
- [distributions::bernoulli::Parameter](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distributions/bernoulli/bernoulli_types.h#L65)

- [distributions::normal::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distributions/normal/normal.h#L89)
- [distributions::normal::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distributions/normal/normal.h#L56)
- [distributions::normal::Parameter](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distributions/normal/normal_types.h#L66)

- [distributions::uniform::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distributions/uniform/uniform.h#L89)
- [distributions::uniform::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distributions/uniform/uniform.h#L56)
- [distributions::uniform::Parameter](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/distributions/uniform/uniform_types.h#L65)

#### [Engines](https://uxlfoundation.github.io/oneDAL/daal/algorithms/engines/index.html)

- [engines::Input](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/engines/engine_types.h#L74)
- [engines::Result](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/engines/engine_types.h#L114)

- [engines::mcg59::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/engines/mcg59/mcg59.h#L56)
- [engines::mcg59::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/engines/mcg59/mcg59.h#L89)

- [engines::mrg32k3a::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/engines/mrg32k3a/mrg32k3a.h#L57)
- [engines::mrg32k3a::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/engines/mrg32k3a/mrg32k3a.h#L90)

- [engines::mt19937::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/engines/mt19937/mt19937.h#L56)
- [engines::mt19937::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/engines/mt19937/mt19937.h#L89)

- [engines::mt2203::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/engines/mt2203/mt2203.h#L56)
- [engines::mt2203::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/engines/mt2203/mt2203.h#L89)

- [engines::philox4x32x10::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/engines/philox4x32x10/philox4x32x10.h#L57)
- [engines::philox4x32x10::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/engines/philox4x32x10/philox4x32x10.h#L90)

#### [Kernel Functions](https://uxlfoundation.github.io/oneDAL/daal/algorithms/kernel_function/kernel-functions.html)

- [kernel_function::Input](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/kernel_function/kernel_function_types.h#L103)
- [kernel_function::Result](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/kernel_function/kernel_function_types.h#L135)

- [kernel_function::linear::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/kernel_function/kernel_function_linear.h#L58)
- [kernel_function::linear::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/kernel_function/kernel_function_linear.h#L92)
- [kernel_function::linear::Input](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/kernel_function/kernel_function_types_linear.h#L83)

- [kernel_function::rbf::BatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/kernel_function/kernel_function_rbf.h#L57)
- [kernel_function::rbf::Batch](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/kernel_function/kernel_function_rbf.h#L91)
- [kernel_function::rbf::Input](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/kernel_function/kernel_function_types_rbf.h#L82)

#### Other APIs

- [AlgorithmDispatchContainer](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/algorithm_container_base_common.h#L66)
- [AlgorithmDispatchContainer<batch, ...>](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/algorithms/algorithm_container_base_batch.h#L164)
