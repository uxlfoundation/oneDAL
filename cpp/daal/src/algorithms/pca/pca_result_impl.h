/* file: pca_result_impl.h */
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
//  Implementation of PCA algorithm result.
//--
*/

#ifndef __PCA_RESULT_IMPL_H__
#define __PCA_RESULT_IMPL_H__

#include "data_management/data/data_collection.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface3
{
class ResultImpl : public data_management::DataCollection
{
public:
    DAAL_CAST_OPERATOR(ResultImpl);

    bool isWhitening;
    ResultImpl(const size_t n) : DataCollection(n), isWhitening(false) {}
    ResultImpl(const ResultImpl & o) : DataCollection(o), isWhitening(o.isWhitening) {}
    ResultImpl & operator=(const ResultImpl & o);
    virtual ~ResultImpl() {};

    /**
    * Allocates memory for storing partial results of the PCA algorithm
    * \param[in] input Pointer to an object containing input data
    * \param[in] nComponents Number of components
    * \param[in] resultsToCompute Results to compute
    * \return Status of computations
    */
    template <typename algorithmFPType>
    services::Status allocate(const daal::algorithms::Input * input, size_t nComponents, DAAL_UINT64 resultsToCompute);

    /**
    * Allocates memory for storing partial results of the PCA algorithm
    * \param[in] partialResult Pointer to an object containing partialResult data
    * \param[in] nComponents Number of components
    * \param[in] resultsToCompute Results to compute
    * \return Status of computations
    */
    template <typename algorithmFPType>
    services::Status allocate(const daal::algorithms::PartialResult * partialResult, size_t nComponents, DAAL_UINT64 resultsToCompute);

    /**
    * Checks the results of the PCA algorithm implementation
    * \param[in] nFeatures      Number of features
    * \param[in] nComponents Number of components
    * \param[in] nTables        Number of tables
    *
    * \return Status
    */
    virtual services::Status check(size_t nFeatures, size_t nComponents, size_t nTables) const;

    /**
    * Checks the results of the PCA algorithm implementation
    * \param[in] nFeatures      Number of features
    * \param[in] nTables        Number of tables
    *
    * \return Status
    */
    virtual services::Status check(size_t nFeatures, size_t nTables) const;

    void setTable(size_t key, data_management::NumericTablePtr table);

protected:
    /**
    * Allocates memory for storing partial results of the PCA algorithm
    * \param[in] nFeatures Number of features
    * \param[in] nComponents Number of components
    * \param[in] resultsToCompute Results to compute
    * \return Status of computations
    */
    template <typename algorithmFPType>
    services::Status allocate(size_t nFeatures, size_t nComponents, DAAL_UINT64 resultsToCompute);

    template <typename algorithmFPType, typename NumericTableType>
    services::Status __allocate__impl(size_t nFeatures, size_t nComponents, DAAL_UINT64 resultsToCompute);
};

} // namespace interface3
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
