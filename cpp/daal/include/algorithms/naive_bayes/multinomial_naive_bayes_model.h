/* file: multinomial_naive_bayes_model.h */
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
//  Implementation of class defining Naive Bayes model.
//--
*/

#ifndef __NAIVE_BAYES_MODEL_H__
#define __NAIVE_BAYES_MODEL_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/classifier/classifier_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup multinomial_naive_bayes Naive Bayes Classifier
 * \copydoc daal::algorithms::multinomial_naive_bayes
 * @ingroup classification
 * @{
 */
/**
 * \brief Contains classes for multinomial Naive Bayes algorithm
 */
namespace multinomial_naive_bayes
{
/**
 * \brief Contains version 2.0 of oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PARAMETER"></a>
 * \brief Naive Bayes algorithm parameters
 *
 * \snippet naive_bayes/multinomial_naive_bayes_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public classifier::Parameter
{
    /**
     *  Main constructor
     *  \param[in] nClasses             Number of classes
     *  \param[in] priorClassEstimates_ Prior class estimates, numeric table of size [nClasses x 1]
     *  \param[in] alpha_               Imagined occurrences of the each feature, numeric table of size [1 x nFeatures]
     */
    Parameter(size_t nClasses, const data_management::NumericTablePtr & priorClassEstimates_ = data_management::NumericTablePtr(),
              const data_management::NumericTablePtr & alpha_ = data_management::NumericTablePtr())
        : classifier::Parameter(nClasses), priorClassEstimates(priorClassEstimates_), alpha(alpha_)
    {}

    data_management::NumericTablePtr priorClassEstimates; /*!< Prior class estimates */
    data_management::NumericTablePtr alpha;               /*!< Imagined occurrences of the each word */

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */
} // namespace interface2

namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__MODEL"></a>
 * \brief Multinomial naive Bayes model
 */
class DAAL_EXPORT Model : public classifier::Model
{
public:
    DECLARE_MODEL(Model, classifier::Model);

    /**
     * Empty constructor for deserialization
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    Model() {}

    /**
     * Constructs multinomial naive Bayes model
     * \param[in] nFeatures  The number of features
     * \param[in] parameter  The multinomial naive Bayes parameter
     * \param[in] dummy      Dummy variable for the templated constructor
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    template <typename modelFPType>
    DAAL_EXPORT Model(size_t nFeatures, const interface2::Parameter & parameter, modelFPType dummy);

    /**
     * Constructs multinomial naive Bayes model
     * \param[in] nFeatures  The number of features
     * \param[in] parameter  The multinomial naive Bayes parameter
     * \param[out] stat      Status of the model construction
     */
    template <typename modelFPType>
    DAAL_EXPORT static services::SharedPtr<Model> create(size_t nFeatures, const interface2::Parameter & parameter, services::Status * stat = NULL);
    /** \private */
    virtual ~Model() {}

    /**
     * Returns a pointer to the Numeric Table with logarithms of priors
     *  \return Pointer to the Numeric Table with logarithms of priors
     */
    data_management::NumericTablePtr getLogP() { return _logP; }

    /**
     * Returns a pointer to the Numeric Table with logarithms of the conditional probabilities
     *  \return Pointer to the Numeric Table with logarithms of the conditional probabilities
     */
    data_management::NumericTablePtr getLogTheta() { return _logTheta; }

    /**
     * Returns a pointer to the Numeric Table with logarithms of the conditional probabilities
     *  \return Pointer to the Numeric Table with logarithms of the conditional probabilities
     */
    data_management::NumericTablePtr getAuxTable() { return _auxTable; }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return (_logTheta ? _logTheta->getNumberOfColumns() : 0); }

protected:
    data_management::NumericTablePtr _logP;
    data_management::NumericTablePtr _logTheta;
    data_management::NumericTablePtr _auxTable;

    template <typename modelFPType>
    DAAL_EXPORT Model(size_t nFeatures, const interface2::Parameter & parameter, modelFPType dummy, services::Status & st);

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        services::Status st = classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        if (!st) return st;
        arch->setSharedPtrObj(_logP);
        arch->setSharedPtrObj(_logTheta);
        arch->setSharedPtrObj(_auxTable);

        return st;
    }
};

typedef services::SharedPtr<Model> ModelPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PARTIALMODEL"></a>
 * \brief PartialModel represents partial multinomial naive Bayes model
 */
class DAAL_EXPORT PartialModel : public classifier::Model
{
public:
    DECLARE_SERIALIZABLE_CAST(PartialModel)
    /**
     * Empty constructor for deserialization
     * \DAAL_DEPRECATED_USE{ PartialModel::create }
     */
    PartialModel();

    /**
     * Constructs multinomial naive Bayes partial model
     * \param[in] nFeatures  The number of features
     * \param[in] parameter  Multinomial naive Bayes parameter
     * \param[in] dummy      Dummy variable for the templated constructor
     * \DAAL_DEPRECATED_USE{ PartialModel::create }
     */
    template <typename modelFPType>
    DAAL_EXPORT PartialModel(size_t nFeatures, const interface2::Parameter & parameter, modelFPType dummy);

    /**
     * Constructs multinomial naive Bayes partial model
     * \param[in] nFeatures  The number of features
     * \param[in] parameter  The multinomial naive Bayes parameter
     * \param[out] stat      Status of the model construction
     * \return Multinomial naive Bayes partial model
     */
    template <typename modelFPType>
    DAAL_EXPORT static services::SharedPtr<PartialModel> create(size_t nFeatures, const interface2::Parameter & parameter,
                                                                services::Status * stat = NULL);
    /** \private */
    virtual ~PartialModel() {}

    size_t getNObservations() { return _nObservations; }

    void setNObservations(size_t nObservations) { _nObservations = nObservations; }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return (_classGroupSum ? _classGroupSum->getNumberOfColumns() : 0); }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNFeatures() const DAAL_C11_OVERRIDE { return getNumberOfFeatures(); }

    /**
     *  Sets the number of features in the dataset was used on the training stage
     *  \param[in]  nFeatures  Number of features in the dataset was used on the training stage
     */
    void setNFeatures(size_t /*nFeatures*/) DAAL_C11_OVERRIDE {}

    template <typename modelFPType>
    services::Status initialize()
    {
        _classSize->assign((int)0);
        _classGroupSum->assign((int)0);
        _nObservations = 0;
        return services::Status();
    }

    data_management::NumericTablePtr getClassSize() { return _classSize; }
    data_management::NumericTablePtr getClassGroupSum() { return _classGroupSum; }

protected:
    data_management::NumericTablePtr _classSize;
    data_management::NumericTablePtr _classGroupSum;
    size_t _nObservations;

    template <typename modelFPType>
    DAAL_EXPORT PartialModel(size_t nFeatures, const interface2::Parameter & parameter, modelFPType dummy, services::Status & st);

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        services::Status st = classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        if (!st) return st;
        arch->set(_nObservations);
        arch->setSharedPtrObj(_classSize);
        arch->setSharedPtrObj(_classGroupSum);

        return st;
    }
};
typedef services::SharedPtr<PartialModel> PartialModelPtr;
} // namespace interface1
using interface2::Parameter;
using interface1::Model;
using interface1::ModelPtr;
using interface1::PartialModel;
using interface1::PartialModelPtr;

} // namespace multinomial_naive_bayes
/** @} */
} // namespace algorithms
} // namespace daal
#endif
