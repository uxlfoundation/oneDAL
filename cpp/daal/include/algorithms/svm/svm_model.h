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

    services::Status check() const DAAL_C11_OVERRIDE;
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
    DECLARE_MODEL(Model, classifier::Model);

    /**
     * Constructs the SVM model
     * \tparam modelFPType  Data type to store SVM model data, double or float
     * \param[in] dummy     Dummy variable for the templated constructor
     * \param[in] nColumns  Number of features in input data
     * \param[in] layout    Data layout of the numeric table of support vectors
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    template <typename modelFPType>
    Model(modelFPType dummy, size_t nColumns, data_management::NumericTableIface::StorageLayout layout = data_management::NumericTableIface::aos)
        : _bias(0.0)
    {
        using namespace data_management;
        if (layout == NumericTableIface::csrArray)
        {
            modelFPType * dummyPtr = NULL;
            _SV.reset(new CSRNumericTable(dummyPtr, NULL, NULL, nColumns));
        }
        else
        {
            _SV.reset(new HomogenNumericTable<modelFPType>(NULL, nColumns, 0));
        }
        _SVCoeff.reset(new HomogenNumericTable<modelFPType>(NULL, 1, 0));
        _SVIndices.reset(new HomogenNumericTable<int>(NULL, 1, 0));
    }

    /**
     * Constructs the SVM model
     * \tparam modelFPType  Data type to store SVM model data, double or float
     * \param[in] nColumns  Number of features in input data
     * \param[in] layout    Data layout of the numeric table of support vectors
     * \param[out] stat     Status of the model construction
     * \return SVM model
     */
    template <typename modelFPType>
    DAAL_EXPORT static services::SharedPtr<Model> create(
        size_t nColumns, data_management::NumericTableIface::StorageLayout layout = data_management::NumericTableIface::aos,
        services::Status * stat = NULL);

    /**
     * Empty constructor for deserialization
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    Model() : _SV(), _SVCoeff(), _bias(0.0), _SVIndices() {}

    /**
     * Constructs empty SVM model for deserialization
     * \param[out] stat     Status of the model construction
     * \return Empty SVM model for deserialization
     */
    static services::SharedPtr<Model> create(services::Status * stat = NULL)
    {
        services::SharedPtr<Model> modelPtr(new Model());
        if (!modelPtr)
        {
            if (stat) stat->add(services::ErrorMemoryAllocationFailed);
        }
        return modelPtr;
    }

    virtual ~Model() {}

    /**
     * Returns support vectors constructed during the training of the SVM model
     * \return Array of support vectors
     */
    data_management::NumericTablePtr getSupportVectors() { return _SV; }

    /**
     * Returns indices of the support vectors constructed during the training of the SVM model
     * \return Array of support vectors indices
     */
    data_management::NumericTablePtr getSupportIndices() { return _SVIndices; }

    /**
     * Returns classification coefficients constructed during the training of the SVM model
     * \return Array of classification coefficients
     */
    data_management::NumericTablePtr getClassificationCoefficients() { return _SVCoeff; }

    /**
     * Returns the bias constructed during the training of the SVM model
     * \return Bias
     */
    virtual double getBias() { return _bias; }

    /**
     * Sets the bias for the SVM model
     * \param bias  Bias of the model
     */
    virtual void setBias(double bias) { _bias = bias; }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return (_SV ? _SV->getNumberOfColumns() : 0); }

protected:
    data_management::NumericTablePtr _SV;        /*!< \private Support vectors */
    data_management::NumericTablePtr _SVCoeff;   /*!< \private Classification coefficients */
    double _bias;                                /*!< \private Bias of the distance function D(x) = w*Phi(x) + bias */
    data_management::NumericTablePtr _SVIndices; /*!< \private Indices of the support vectors in training data set */

    template <typename modelFPType>
    DAAL_EXPORT Model(modelFPType dummy, size_t nColumns, data_management::NumericTableIface::StorageLayout layout, services::Status & st);

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        services::Status st = classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        if (!st) return st;
        arch->setSharedPtrObj(_SV);
        arch->setSharedPtrObj(_SVCoeff);
        arch->set(_bias);

        arch->setSharedPtrObj(_SVIndices);

        return st;
    }
};
typedef services::SharedPtr<Model> ModelPtr;
/** @} */
} // namespace interface1
using interface2::Parameter;
using interface1::Model;
using interface1::ModelPtr;

} // namespace svm
/** @} */
} // namespace algorithms
} // namespace daal
#endif
