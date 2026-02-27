/* file: implicit_als_train_init_parameter.cpp */
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
//  Implementation of auxiliary implicit als methods.
//--
*/

#include "src/algorithms/implicit_als/implicit_als_train_init_parameter.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace internal
{
SharedPtr<HomogenNumericTable<int> > getPartition(const init::DistributedParameter * parameter, services::Status & st)
{
    NumericTable * partitionTable = parameter->partition.get();
    size_t nRows                  = partitionTable->getNumberOfRows();
    size_t nParts                 = nRows - 1;
    BlockDescriptor<int> block;
    if (nRows == 1)
    {
        partitionTable->getBlockOfRows(0, nRows, readOnly, block);
        nParts = *(block.getBlockPtr());
        partitionTable->releaseBlockOfRows(block);
    }
    SharedPtr<HomogenNumericTable<int> > nt = HomogenNumericTable<int>::create(1, nParts + 1, NumericTable::doAllocate, st);
    if (!st) return nt;

    int * partition = nt->getArray();
    if (nRows == 1)
    {
        size_t nUsersInPart = parameter->fullNUsers / nParts;
        partition[0]        = 0;
        for (size_t i = 1; i < nParts; i++)
        {
            partition[i] = partition[i - 1] + nUsersInPart;
        }
        partition[nParts] = parameter->fullNUsers;
    }
    else
    {
        partitionTable->getBlockOfRows(0, nRows, readOnly, block);
        int * srcPartition = block.getBlockPtr();
        for (size_t i = 0; i < nParts + 1; i++)
        {
            partition[i] = srcPartition[i];
        }
        partitionTable->releaseBlockOfRows(block);
    }
    return nt;
}

} // namespace internal
} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
