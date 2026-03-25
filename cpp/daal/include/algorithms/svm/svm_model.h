/* file: svm_model.h */
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
//  Implementation of the class defining the SVM model.
//--
*/

#ifndef __SVM_MODEL_H__
#define __SVM_MODEL_H__

#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "algorithms/model.h"
#include "algorithms/kernel_function/kernel_function.h"
#include "algorithms/kernel_function/kernel_function_linear.h"
#include "algorithms/kernel_function/kernel_function_types.h"
#include "algorithms/classifier/classifier_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup svm Support Vector Machine Classifier
 * \copydoc daal::algorithms::svm
 * @ingroup classification
 * @{
 */
/**
 * \brief Contains classes to work with the support vector machine classifier
 */
namespace svm
{
/**
 * \brief Contains version 2.0 of oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * @ingroup svm
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__SVM__PARAMETER"></a>
 * \brief Optional parameters
 *
 * \snippet svm/svm_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public classifier::Parameter
{
    Parameter(const services::SharedPtr<kernel_function::KernelIface> & kernelForParameter =
                  services::SharedPtr<kernel_function::KernelIface>(new kernel_function::linear::Batch<>()),
              double C = 1.0, double accuracyThreshold = 0.001, double tau = 1.0e-6, size_t maxIterations = 1000000, size_t cacheSize = 8000000,
              bool doShrinking = true, size_t shrinkingStep = 1000)
        : C(C),
          accuracyThreshold(accuracyThreshold),
          tau(tau),
          maxIterations(maxIterations),
          cacheSize(cacheSize),
          doShrinking(doShrinking),
          shrinkingStep(shrinkingStep),
          kernel(kernelForParameter) {};

    double C;                                           /*!< Upper bound in constraints of the quadratic optimization problem */
    double accuracyThreshold;                           /*!< Training accuracy */
    double tau;                                         /*!< Tau parameter of the working set selection scheme */
    size_t maxIterations;                               /*!< Maximal number of iterations for the algorithm */
    size_t cacheSize;                                   /*!< Size of cache in bytes to store values of the kernel matrix.
                                     A non-zero value enables use of a cache optimization technique */
    bool doShrinking;                                   /*!< Flag that enables use of the shrinking optimization technique */
    size_t shrinkingStep;                               /*!< Number of iterations between the steps of shrinking optimization technique */
    algorithms::kernel_function::KernelIfacePtr kernel; /*!< Kernel function */

    services::Status check() const override;
};
/* [Parameter source code] */
} // namespace interface2

namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__MODEL"></a>
 * \brief %Model of the classifier trained by the svm::training::Batch algorithm
 *
 * \par References
 *      - Parameter class
 *      - \ref training::interface2::Batch "training::Batch" class
 *      - \ref prediction::interface2::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public classifier::Model
{
public:
    DECLARE_MODEL(Model, classifier::Model)

    virtual ~Model() {}

    /**
     * Returns support vectors constructed during the training of the SVM model
     * \return Array of support vectors
     */
    virtual data_management::NumericTablePtr getSupportVectors() const = 0;

    /**
     * Returns indices of the support vectors constructed during the training of the SVM model
     * \return Array of support vectors indices
     */
    virtual data_management::NumericTablePtr getSupportIndices() const = 0;

    /**
     * Returns classification coefficients constructed during the training of the SVM model
     * \return Array of classification coefficients
     */
    virtual data_management::NumericTablePtr getClassificationCoefficients() const = 0;

    /**
     * Returns the bias constructed during the training of the SVM model
     * \return Bias
     */
    virtual double getBias() const = 0;

    /**
     * Returns biases constructed during the training of the two-class or multi-class SVM model
     * \return Array of biases
     */
    virtual data_management::NumericTablePtr getBiases() const = 0;

    /**
     * Returns the number of iterations performed during the training of the SVM model
     * \return Number of iterations
     */
    virtual size_t getNumberOfIterations() const = 0;
};
typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;
/** @} */
} // namespace interface1
using interface2::Parameter;
using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;

} // namespace svm
/** @} */
} // namespace algorithms
} // namespace daal
#endif
