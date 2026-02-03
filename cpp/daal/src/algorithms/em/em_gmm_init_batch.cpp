/* file: em_gmm_init_batch.cpp */
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
//  Implementation of EMforKernel
//--
*/

#include "algorithms/em/em_gmm_init_types.h"
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
namespace init
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_EM_GMM_INIT_RESULT_ID);

Parameter::Parameter(size_t nComponents, size_t nTrials, size_t nIterations, size_t seed, double accuracyThreshold,
                     em_gmm::CovarianceStorageId covarianceStorage)
    : nComponents(nComponents),
      nTrials(nTrials),
      nIterations(nIterations),
      seed(seed),
      accuracyThreshold(accuracyThreshold),
      covarianceStorage(covarianceStorage),
      engine(engines::mt19937::Batch<>::create())
{}

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter::check() const
{
    DAAL_CHECK_EX(accuracyThreshold >= 0, ErrorEMInitIncorrectToleranceToConverge, ParameterName, accuracyThresholdStr());
    DAAL_CHECK_EX(nIterations > 0, ErrorEMInitIncorrectDepthNumberIterations, ParameterName, nIterationsStr());
    DAAL_CHECK_EX(nTrials > 0, ErrorEMInitIncorrectNumberOfTrials, ParameterName, nTrialsStr());
    DAAL_CHECK_EX(nComponents > 0, ErrorEMInitIncorrectNumberOfComponents, ParameterName, nComponentsStr());
    return services::Status();
}

/** Default constructor */
Input::Input() : daal::algorithms::Input(lastInputId + 1) {}

/**
* Sets the input for the EM for GMM algorithm
* \param[in] id    Identifier of the input
* \param[in] ptr   Pointer to the value
*/
void Input::set(InputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
* Returns the input NumericTable for the computation of initial values for the EM for GMM algorithm
* \param[in] id    Identifier of the input NumericTable
* \return          %Input NumericTable that corresponds to the given identifier
*/
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
* Checks input for the computation of initial values for the EM for GMM algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Method of the algorithm
*/
services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    return checkNumericTable(get(data).get(), dataStr());
}

/** Default constructor */
Result::Result() : daal::algorithms::Result(lastResultCovariancesId + 1)
{
    Argument::set(covariances, DataCollectionPtr(new DataCollection()));
}

/**
 * Sets the result for the computation of initial values for the EM for GMM algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the numeric table for the result
 */
void Result::set(ResultId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the covariance collection for initialization of EM for GMM algorithm
 * \param[in] id    Identifier of the collection of covariances
 * \param[in] ptr   Pointer to the collection of covariances
 */
void Result::set(ResultCovariancesId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the result for the computation of initial values for the EM for GMM algorithm
 * \param[in] id   %Result identifier
 * \return         %Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Returns the collection of initialized covariances for the EM for GMM algorithm
 * \param[in] id   Identifier of the collection of covariances
 * \return         Collection of covariances that corresponds to the given identifier
 */
DataCollectionPtr Result::get(ResultCovariancesId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

/**
 * Returns the covariance with a given index from the collection of initialized covariances
 * \param[in] id    Identifier of the collection of covariances
 * \param[in] index Index of the covariance to be returned
 * \return          Pointer to the table with initialized covariances
 */
NumericTablePtr Result::get(ResultCovariancesId id, size_t index) const
{
    DataCollectionPtr covCollection = this->get(id);
    return staticPointerCast<NumericTable, SerializationIface>((*covCollection)[index]);
}

/**
* Checks the result of the computation of initial values for the EM for GMM algorithm
* \param[in] input   %Input of the algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Method of the algorithm
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
        if (algParameter->covarianceStorage == em_gmm::full)
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
} // namespace init
} // namespace em_gmm
} // namespace algorithms
} // namespace daal
