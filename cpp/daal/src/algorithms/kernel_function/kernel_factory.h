#ifndef __KERNEL_FACTORY_H__
#define __KERNEL_FACTORY_H__

#include "algorithms/kernel_function/kernel_function.h"
#include "src/algorithms/kernel_function/kernel_function_linear.h"
#include "src/algorithms/kernel_function/kernel_function_rbf.h"
#include "src/algorithms/kernel_function/polynomial/kernel_function_polynomial.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{

inline KernelIfacePtr createKernel(KernelType type)
{
    switch (type)
    {
    case KernelType::linear: return KernelIfacePtr(new linear::Batch<>());
    case KernelType::rbf: return KernelIfacePtr(new rbf::Batch<>());
    case KernelType::polynomial: return KernelIfacePtr(new polynomial::Batch<>());
    default: return KernelIfacePtr(new linear::Batch<>());
    }
}

} // namespace kernel_function
} // namespace algorithms
} // namespace daal
#endif
