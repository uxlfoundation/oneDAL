/* file: em_gmm_batch.cpp */
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
//  Implementation of the EM for GMM algorithm interface.
//--
*/

#include "algorithms/em/em_gmm_types.h"
#include "services/daal_defines.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_EM_GMM_RESULT_ID);
/**
 * Constructs the parameter of EMM for GMM algorithm
 * \param[in] nComponents              Number of components in the Gaussian mixture model
 * \param[in] maxIterations            Maximal number of iterations of the algorithm
 * \param[in] accuracyThreshold        Threshold for the termination of the algorithm
 * \param[in] covariance               Pointer to the algorithm that computes the covariance
 */
Parameter::Parameter(const size_t _nComponents, const SharedPtr<covariance::BatchImpl> & _covariance, const size_t _maxIterations,
                     const double _accuracyThreshold, const double _regularizationFactor, const CovarianceStorageId _covarianceStorage)
    : nComponents(_nComponents),
      maxIterations(_maxIterations),
      accuracyThreshold(_accuracyThreshold),
      covariance(_covariance),
      regularizationFactor(_regularizationFactor),
      covarianceStorage(_covarianceStorage)
{}

Parameter::Parameter(const Parameter & other)
    : nComponents(other.nComponents),
      maxIterations(other.maxIterations),
      accuracyThreshold(other.accuracyThreshold),
      covariance(other.covariance),
      regularizationFactor(other.regularizationFactor),
      covarianceStorage(other.covarianceStorage)
{}

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(accuracyThreshold >= 0, ErrorEMIncorrectToleranceToConverge, ParameterName, accuracyThresholdStr());
    DAAL_CHECK_EX(maxIterations > 0, ErrorEMIncorrectMaxNumberOfIterations, ParameterName, maxIterationsStr());
    DAAL_CHECK_EX(nComponents > 0, ErrorEMIncorrectNumberOfComponents, ParameterName, nComponentsStr());
    DAAL_CHECK_EX(covariance, ErrorNullAuxiliaryAlgorithm, ParameterName, covarianceStr());
    DAAL_CHECK(regularizationFactor >= 0, ErrorIncorrectParameter);
    return services::Status();
}

/** Default constructor */
Input::Input() : daal::algorithms::Input(lastInputValuesId + 1) {}

/**
 * Sets one input object for the EM for GMM algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the input covariance object for the EM for GMM algorithm
 * \param[in] id    Identifier of the input covariance collection object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputCovariancesId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets input objects for the EM for GMM algorithm[IE1]
 * \param[in] id    Identifier of the input values object. Result of the EM for GMM initialization algorithm can be used.
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputValuesId id, const init::ResultPtr & ptr)
{
    if (ptr)
    {
        set(inputWeights, ptr->get(init::weights));
        set(inputMeans, ptr->get(init::means));
        set(inputCovariances, ptr->get(init::covariances));
    }
}

/**
 * Returns the input numeric table for the EM for GMM algorithm
 * \param[in] id    Identifier of the input numeric table
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Returns the collection of input covariances for the EM for GMM algorithm
 * \param[in] id    Identifier of the  collection of input covariances
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr Input::get(InputCovariancesId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

/**
 * Returns a covariance with a given index from the collection of input covariances
 * \param[in] id    Identifier of the collection of input covariances
 * \param[in] index Index of the covariance to be returned
 * \return          Pointer to the table with the input covariance
 */
NumericTablePtr Input::get(InputCovariancesId id, size_t index) const
{
    DataCollectionPtr covCollection = this->get(id);
    return staticPointerCast<NumericTable, SerializationIface>((*covCollection)[index]);
}

/**
 * Checks the correctness of the input result
 * \param[in] par       Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    services::Status s;
    s |= checkNumericTable(get(data).get(), dataStr());
    if (!s) return s;

    Parameter * algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
    size_t nComponents       = algParameter->nComponents;
    size_t nFeatures         = get(data)->getNumberOfColumns();

    s |= checkNumericTable(get(inputWeights).get(), inputWeightsStr(), 0, 0, nComponents, 1);
    if (!s) return s;
    s |= checkNumericTable(get(inputMeans).get(), inputMeansStr(), 0, 0, nFeatures, nComponents);
    if (!s) return s;

    DataCollectionPtr inputCovCollection = get(inputCovariances);
    DAAL_CHECK(inputCovCollection, ErrorNullInputDataCollection);
    DAAL_CHECK(inputCovCollection->size() == nComponents, ErrorIncorrectNumberOfInputNumericTables);

    for (size_t i = 0; i < nComponents; i++)
    {
        SerializationIfacePtr collectionElement = (*inputCovCollection)[i];
        DAAL_CHECK_EX(collectionElement, ErrorNullNumericTable, ArgumentName, inputCovariancesStr());

        NumericTablePtr nt = NumericTable::cast(collectionElement);
        DAAL_CHECK_EX(nt, ErrorIncorrectElementInCollection, ArgumentName, inputCovariancesStr());

        if (algParameter->covarianceStorage == full)
        {
            s |= checkNumericTable(nt.get(), inputCovariancesStr(), 0, 0, nFeatures, nFeatures);
            if (!s) return s;
        }
        else
        {
            s |= checkNumericTable(nt.get(), inputCovariancesStr(), 0, 0, nFeatures, 1);
        }
    }
    return s;
}

/** Default constructor */
Result::Result() : daal::algorithms::Result(lastResultCovariancesId + 1)
{
    Argument::set(covariances, DataCollectionPtr(new DataCollection()));
}

/**
 * Sets the result of the EM for GMM algorithm
 * \param[in] id    %Result identifier
 * \param[in] ptr   Pointer to the numeric table with the result
 */
void Result::set(ResultId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the collection of covariances for the EM for GMM algorithm
 * \param[in] id    Identifier of the collection of covariances
 * \param[in] ptr   Pointer to the collection of covariances
 */
void Result::set(ResultCovariancesId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the result of the EM for GMM algorithm
 * \param[in] id   %Result identifier
 * \return         %Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Returns the collection of computed covariances of the EM for GMM algorithm
 * \param[in] id   Identifier of the collection of computed covariances
 * \return         Collection of computed covariances that corresponds to the given identifier
 */
DataCollectionPtr Result::get(ResultCovariancesId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

/**
 * Returns the covariance with a given index from the collection of computed covariances
 * \param[in] id    Identifier of the collection of covariances
 * \param[in] index Index of the covariance to be returned
 * \return          Pointer to the table with the computed covariance
 */
NumericTablePtr Result::get(ResultCovariancesId id, size_t index) const
{
    DataCollectionPtr covCollection = this->get(id);
    return staticPointerCast<NumericTable, SerializationIface>((*covCollection)[index]);
}

/**
* Checks the result parameter of the EM for GMM algorithm
* \param[in] input   %Input of the algorithm
* \param[in] par     %Parameter of algorithm
* \param[in] method  Computation method
*/
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    Input * algInput         = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    Parameter * algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));

    size_t nComponents = algParameter->nComponents;
    size_t nFeatures   = algInput->get(data)->getNumberOfColumns();

    services::Status s;
    int unexpectedLayouts = packed_mask;
    s |= checkNumericTable(get(weights).get(), weightsStr(), unexpectedLayouts, 0, nComponents, 1);
    if (!s) return s;
    s |= checkNumericTable(get(means).get(), meansStr(), unexpectedLayouts, 0, nFeatures, nComponents);
    if (!s) return s;

    int unexpectedLayoutCSR = (int)NumericTableIface::csrArray;
    s |= checkNumericTable(get(goalFunction).get(), goalFunctionStr(), unexpectedLayoutCSR, 0, 1, 1);
    if (!s) return s;
    s |= checkNumericTable(get(nIterations).get(), nIterationsStr(), unexpectedLayoutCSR, 0, 1, 1);
    if (!s) return s;

    DataCollectionPtr resultCovCollection = get(covariances);
    DAAL_CHECK(resultCovCollection, ErrorNullOutputDataCollection);
    DAAL_CHECK(resultCovCollection->size() == nComponents, ErrorIncorrectNumberOfOutputNumericTables);

    constexpr int unexpectedLayoutCovariance = (int)NumericTableIface::csrArray;
    for (size_t i = 0; i < nComponents; i++)
    {
        SerializationIfacePtr collectionElement = (*resultCovCollection)[i];
        DAAL_CHECK_EX(collectionElement, ErrorNullNumericTable, ArgumentName, covarianceStr());

        NumericTablePtr nt = NumericTable::cast(collectionElement);
        DAAL_CHECK_EX(nt, ErrorIncorrectElementInCollection, ArgumentName, covarianceStr());
        if (algParameter->covarianceStorage == full)
        {
            s |= checkNumericTable(nt.get(), covarianceStr(), unexpectedLayoutCovariance, 0, nFeatures, nFeatures);
            if (!s) return s;
        }
        else
        {
            s |= checkNumericTable(nt.get(), covarianceStr(), unexpectedLayoutCovariance, 0, nFeatures, 1);
        }
    }
    return s;
}

} // namespace interface1
} // namespace em_gmm
} // namespace algorithms
} // namespace daal
