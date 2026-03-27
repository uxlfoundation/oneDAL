/* file: svm_model_impl.h */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#ifndef __SVM_MODEL_IMPL_H__
#define __SVM_MODEL_IMPL_H__

#include "algorithms/svm/svm_model.h"
#include "src/algorithms/classifier/classifier_model_impl.h"
#include "src/services/serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace internal
{
using namespace data_management;

/**
 * \brief Class that implements the methods for models trained with the SVM algorithm
 */
class ModelInternal
{
public:
    /**
     * Constructs the SVM model implemenatation
     * \param[in] dummy    Data type dummy variable for the templated constructor.
     *                     Defines the data type of the model coefficients and support vectors.
     * \param[in] nClasses Number of classes in the training data
     * \param[in] nColumns Number of columns in the training data
     * \param[in] layout   Storage layout of the numeric table with support vectors.
     *                     Provide NumericTableIface::csrArray for sparse tables.
     * \param[out] st      Status of the model construction
     */
    template <typename modelFPType>
    ModelInternal(modelFPType dummy, size_t nClasses, size_t nColumns, data_management::NumericTableIface::StorageLayout layout, services::Status & st);

    ModelInternal();

    virtual ~ModelInternal() {}

    /**
     * Returns support vectors constructed during the training of the SVM model
     * \return Array of support vectors
     */
    data_management::NumericTablePtr getSupportVectors() const;

    void setSupportVectors(data_management::NumericTablePtr & supportVectors);

    /**
     * Returns indices of the support vectors constructed during the training of the SVM model
     * \return Array of support vectors indices
     */
    data_management::NumericTablePtr getSupportIndices() const;

    void setSupportIndices(data_management::NumericTablePtr & supportIndices);

    /**
     * Returns classification coefficients constructed during the training of the SVM model
     * \return Array of classification coefficients
     */
    data_management::NumericTablePtr getClassificationCoefficients() const;

    void setClassificationCoefficients(data_management::NumericTablePtr & classificationCoefficients);

    /**
     * Returns the bias constructed during the training of the SVM model
     * \return Bias
     */
    double getBias() const;

    data_management::NumericTablePtr getBiases() const;

    void setBiases(data_management::NumericTablePtr & biases);

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const;

    /**
     * Returns the number of iterations performed during the training of the SVM model
     * \return Number of iterations
     */
    size_t getNumberOfIterations() const;

    void setNumberOfIterations(size_t nIterations);

protected:
    data_management::NumericTablePtr _SV;        /* Support vectors */
    data_management::NumericTablePtr _SVCoeff;   /* Classification coefficients */
    data_management::NumericTablePtr _biases;    /* Biases of the distance function D(x) = w*Phi(x) + bias
                                                    _biases[i] holds the bias for the i-th model in the multiclass SVM */
    data_management::NumericTablePtr _SVIndices; /* Indices of the support vectors in training data set */
    size_t _nIterations;                         /* Number of iterations performed during the training */
};

/**
 * \brief Class that connects the interface svm::Model and implementation ModelInternal
 */
class ModelImpl : public svm::Model,
                  protected svm::internal::ModelInternal
{
public:
    typedef svm::internal::ModelInternal ImplType;
    typedef algorithms::classifier::internal::ModelImpl ClassifierImplType;

    /**
     * Constructs the SVM model
     * \param[in] dummy    Data type dummy variable for the templated constructor.
     *                     Defines the data type of the model coefficients and support vectors.
     * \param[in] nClasses Number of classes in the training data
     * \param[in] nColumns Number of columns in the training data
     * \param[in] layout   Storage layout of the numeric table with support vectors.
     *                     Provide NumericTableIface::csrArray for sparse tables.
     * \param[out] st      Status of the model construction
     */
    template <typename modelFPType>
    ModelImpl(modelFPType dummy, size_t nClasses, size_t nColumns, data_management::NumericTableIface::StorageLayout layout, services::Status & st)
        : ImplType(dummy, nClasses, nColumns, layout, st)
    {}

    /**
    * Empty constructor for deserialization
    */
    ModelImpl() : ImplType() {}

    /**
     * Returns support vectors constructed during the training of the SVM model
     * \return Array of support vectors
     */
    data_management::NumericTablePtr getSupportVectors() const override { return ImplType::getSupportVectors(); }

    void setSupportVectors(data_management::NumericTablePtr & supportVectors)
    {
        ImplType::setSupportVectors(supportVectors);
    }

    /**
     * Returns indices of the support vectors constructed during the training of the SVM model
     * \return Array of support vectors indices
     */
    data_management::NumericTablePtr getSupportIndices() const override { return ImplType::getSupportIndices(); }

    void setSupportIndices(data_management::NumericTablePtr & supportIndices)
    {
        ImplType::setSupportIndices(supportIndices);
    }

    /**
     * Returns classification coefficients constructed during the training of the SVM model
     * \return Array of classification coefficients
     */
    data_management::NumericTablePtr getClassificationCoefficients() const override { return ImplType::getClassificationCoefficients(); }

    void setClassificationCoefficients(data_management::NumericTablePtr & classificationCoefficients)
    {
        ImplType::setClassificationCoefficients(classificationCoefficients);
    }

    /**
     * Returns the bias constructed during the training of the SVM model
     * \return Bias
     */
    double getBias() const override { return ImplType::getBias(); }

    /**
     * Returns biases constructed during the training of the two-class or multi-class SVM model
     * \return Array of biases
     */
    data_management::NumericTablePtr getBiases() const override { return ImplType::getBiases(); }

    void setBiases(data_management::NumericTablePtr & biases) { ImplType::setBiases(biases); }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const override { return ImplType::getNumberOfFeatures(); }

    /**
     * Returns the number of iterations performed during the training of the SVM model
     * \return Number of iterations
     */
    size_t getNumberOfIterations() const override { return ImplType::getNumberOfIterations(); }

    void setNumberOfIterations(size_t nIterations) { ImplType::setNumberOfIterations(nIterations); }

protected:
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        ClassifierImplType * classifierImpl = dynamic_cast<ClassifierImplType *>(this);
        if (!classifierImpl) return services::Status(services::ErrorIncorrectTypeOfModel);
        services::Status st = classifierImpl->serialImpl<Archive, onDeserialize>(arch);
        if (!st) return st;
        arch->setSharedPtrObj(_SV);
        arch->setSharedPtrObj(_SVCoeff);
        arch->setSharedPtrObj(_biases);
        arch->set(_nIterations);

        arch->setSharedPtrObj(_SVIndices);

        return st;
    }
};

} // namespace internal
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
