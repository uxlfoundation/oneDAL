/* file: kmeans_plusplus_init_impl.i */
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
//  Implementation of K-means++ initialization algorithm.
//--
*/

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_spblas.h"
#include "src/algorithms/distributions/uniform/uniform_kernel.h"
#include "src/services/service_data_utils.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;
using namespace daal::algorithms::distributions::uniform::internal;

static const size_t _nRowsInBlock = 512u;

//BlockHelperXXX template class is a helper class for a block of rows.
//It provides multiplication of the block's rows to the given matrix of centers
//and the result access according to the data type.
//Single interface hides data-specific manipulations.

//BlockHelperDense is the helper class for the dense data type
template <typename algorithmFPType, CpuType cpu, typename NumericTableType>
class BlockHelperDense
{
public:
    BlockHelperDense(NumericTableType * nt, size_t dim, size_t iStartRow, size_t nRowsToProcess) : _ntDataBD(nt, iStartRow, nRowsToProcess), _dim(dim)
    {}
    const Status & status() const { return _ntDataBD.status(); }

    void callGemm(const algorithmFPType * pCenters, size_t nRows, size_t nCenters, algorithmFPType * gemmResult)
    {
        char transa           = 't';
        char transb           = 'n';
        DAAL_INT _m           = nCenters;
        DAAL_INT _n           = nRows;
        DAAL_INT _k           = _dim;
        algorithmFPType alpha = 1.0;
        DAAL_INT lda          = _dim;
        DAAL_INT ldy          = _dim;
        algorithmFPType beta  = 0.0;
        DAAL_INT ldaty        = nCenters;

        BlasInst<algorithmFPType, cpu>::xxgemm(&transa, &transb, &_m, &_n, &_k, &alpha, pCenters, &lda, _ntDataBD.get(), &ldy, &beta, gemmResult,
                                               &ldaty);
    }

    algorithmFPType getRowSumSq(size_t iRow, const algorithmFPType * cen)
    {
        const algorithmFPType * pData = _ntDataBD.get() + iRow * _dim;
        algorithmFPType norm2         = 0;
        for (size_t i = 0; i < _dim; ++i) norm2 += (pData[i] - cen[i]) * (pData[i] - cen[i]);
        return norm2;
    }

    algorithmFPType getGemmResult(size_t iRow, size_t iCol, size_t nRows, size_t nCols, const algorithmFPType * gemmResult) const
    {
        return gemmResult[iRow * nCols + iCol];
    }

protected:
    ReadRows<algorithmFPType, cpu> _ntDataBD;
    const size_t _dim;
};

//BlockHelperCSR is the helper class for the CSR data type
template <typename algorithmFPType, CpuType cpu, typename NumericTableType>
class BlockHelperCSR
{
public:
    BlockHelperCSR(NumericTableType * nt, size_t dim, size_t iStartRow, size_t nRowsToProcess) : _ntDataBD(nt, iStartRow, nRowsToProcess), _dim(dim)
    {}
    const Status & status() const { return _ntDataBD.status(); }

    void callGemm(const algorithmFPType * pCenters, size_t nRows, size_t nCenters, algorithmFPType * gemmResult)
    {
        char transa           = 'n';
        DAAL_INT _n           = nRows;
        DAAL_INT _p           = _dim;
        DAAL_INT _c           = nCenters;
        algorithmFPType alpha = 1.0;
        algorithmFPType beta  = 0.0;
        char matdescra[6]     = { 'G', 0, 0, 'F', 0, 0 };

        const algorithmFPType * pData = _ntDataBD.values();
        const size_t * colIdx         = _ntDataBD.cols();
        const size_t * rowIdx         = _ntDataBD.rows();

        SpBlasInst<algorithmFPType, cpu>::xxcsrmm(&transa, &_n, &_c, &_p, &alpha, matdescra, pData, (const DAAL_INT *)colIdx,
                                                  (const DAAL_INT *)rowIdx, pCenters, &_p, &beta, gemmResult, &_n);
    }

    algorithmFPType getRowSumSq(size_t iRow, const algorithmFPType * cen)
    {
        const size_t * rowIdx         = _ntDataBD.rows();
        const algorithmFPType * pData = _ntDataBD.values() + rowIdx[iRow] - 1;
        const size_t * colIdx         = _ntDataBD.cols() + rowIdx[iRow] - 1;
        const size_t nValues          = rowIdx[iRow + 1] - rowIdx[iRow];
        algorithmFPType res(0.);
        for (size_t i = 0; i < nValues; ++i) res += (pData[i] - cen[colIdx[i] - 1]) * (pData[i] - cen[colIdx[i] - 1]);
        return res;
    }

    algorithmFPType getGemmResult(size_t iRow, size_t iCol, size_t nRows, size_t nCols, const algorithmFPType * gemmResult) const
    {
        return gemmResult[iRow + iCol * nRows];
    }

protected:
    ReadRowsCSR<algorithmFPType, cpu> _ntDataBD;
    const size_t _dim;
};

//DataHelperXXX template class is used by kmeans init tasks to hide data-specific manipulations behind general interface

//DataHelperDense is the helper class for the dense data type
template <typename algorithmFPType, CpuType cpu>
class DataHelperDense
{
public:
    typedef BlockHelperDense<algorithmFPType, cpu, NumericTable> BlockHelperType;

    DataHelperDense(NumericTable * ntData) : dim(ntData->getNumberOfColumns()), nRows(ntData->getNumberOfRows()), _nt(ntData) {}

    NumericTable * nt() const { return _nt; }
    NumericTable * ntIface() const { return _nt; }

    Status updateMinDistInBlock(algorithmFPType * const minDistAccTrials, size_t nBlock, size_t iBlock, size_t nTrials, size_t iBestTrial,
                                const algorithmFPType * aWeights, const algorithmFPType * const pLastAddedCenter, algorithmFPType * const aMinDist)
    {
        const size_t iStartRow      = iBlock * _nRowsInBlock;                                                  //start row
        const size_t nRowsToProcess = (iBlock == nBlock - 1) ? nRows - iBlock * _nRowsInBlock : _nRowsInBlock; //rows to process

        ReadRows<algorithmFPType, cpu> ntDataBD(nt(), iStartRow, nRowsToProcess);
        DAAL_CHECK_BLOCK_STATUS(ntDataBD);
        const algorithmFPType * const pData = ntDataBD.get();

        algorithmFPType * const pDistSqBest   = &aMinDist[iBestTrial * nRows + iStartRow];
        const algorithmFPType * const weights = aWeights ? &aWeights[iStartRow] : nullptr;
        for (size_t iTrials = 0u; iTrials < nTrials; iTrials++)
        {
            if (iTrials == iBestTrial) continue;

            algorithmFPType * const pDistSq            = &aMinDist[iTrials * nRows + iStartRow];
            const algorithmFPType * const pAddedCenter = &pLastAddedCenter[iTrials * dim];

            minDistAccTrials[iTrials * nBlock + iBlock] =
                updateMinDistForITrials(pDistSq, iTrials, nRowsToProcess, pData, pAddedCenter, weights, pDistSqBest);
        }
        minDistAccTrials[iBestTrial * nBlock + iBlock] =
            updateMinDistForITrials(pDistSqBest, iBestTrial, nRowsToProcess, pData, pLastAddedCenter, weights, pDistSqBest);

        return Status();
    }

    algorithmFPType updateMinDistForITrials(algorithmFPType * const pDistSq, size_t iTrials, size_t nRowsToProcess,
                                            const algorithmFPType * const pData, const algorithmFPType * const pLastAddedCenter,
                                            const algorithmFPType * const aWeights, const algorithmFPType * const pDistSqBest)
    {
        algorithmFPType sumOfDist2 = algorithmFPType(0);

        for (size_t iRow = 0u; iRow < nRowsToProcess; iRow++)
        {
            algorithmFPType dist2 = algorithmFPType(0);
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0u; i < dim; i++)
            {
                dist2 += (pData[iRow * dim + i] - pLastAddedCenter[i]) * (pData[iRow * dim + i] - pLastAddedCenter[i]);
            }
            if (aWeights)
            {
                dist2 *= aWeights[iRow];
            }

            pDistSq[iRow] = daal::services::internal::min<cpu, algorithmFPType>(pDistSqBest[iRow], dist2);
            sumOfDist2 += pDistSq[iRow];
        }

        return sumOfDist2;
    }

    //copy one row from the given table to the destination buffer and return the sum of squares
    //of the data in this row
    algorithmFPType copyOneRowCalcSumSq(size_t iRow, algorithmFPType * pDst) const
    {
        ReadRows<algorithmFPType, cpu> ntDataBD(nt(), iRow, 1);
        const algorithmFPType * pData = ntDataBD.get();
        algorithmFPType res(0.);

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < dim; ++i)
        {
            pDst[i] = pData[i];
            res += pData[i] * pData[i];
        }
        return res;
    }

public:
    const size_t dim;
    const size_t nRows;

protected:
    NumericTable * _nt;
};

//DataHelperCSR is the helper class for the CSR data type
template <typename algorithmFPType, CpuType cpu>
class DataHelperCSR
{
public:
    typedef BlockHelperCSR<algorithmFPType, cpu, CSRNumericTable> BlockHelperType;

    DataHelperCSR(NumericTable * ntData)
        : dim(ntData->getNumberOfColumns()), nRows(ntData->getNumberOfRows()), _nt(ntData), _csr(dynamic_cast<CSRNumericTable *>(ntData))
    {}
    NumericTable * nt() const { return _nt; }
    CSRNumericTable * ntIface() const { return _csr; }

    Status updateMinDistInBlock(algorithmFPType * const minDistAccTrials, size_t nBlock, size_t iBlock, size_t nTrials, size_t iBestTrial,
                                const algorithmFPType * aWeights, const algorithmFPType * const pLastAddedCenter, algorithmFPType * const aMinDist)
    {
        const size_t iStartRow      = iBlock * _nRowsInBlock;                                                  //start row
        const size_t nRowsToProcess = (iBlock == nBlock - 1) ? nRows - iBlock * _nRowsInBlock : _nRowsInBlock; //rows to process

        // TODO: Better to use ReadRowsCSR, but there is a bug related to static library linking.
        // Fixme when ReadRowsCSR will be fixed.
        daal::data_management::CSRBlockDescriptor<algorithmFPType> block;
        _csr->getSparseBlock(iStartRow, nRowsToProcess, daal::data_management::readOnly, block);
        const auto pData  = block.getBlockValuesPtr();
        const auto colIdx = block.getBlockColumnIndicesPtr();
        const auto rowIdx = block.getBlockRowIndicesPtr();

        algorithmFPType * const pDistSqBest   = &aMinDist[iBestTrial * nRows + iStartRow];
        const algorithmFPType * const weights = aWeights ? &aWeights[iStartRow] : nullptr;
        for (size_t iTrials = 0u; iTrials < nTrials; iTrials++)
        {
            if (iTrials == iBestTrial) continue;

            algorithmFPType * const pDistSq            = &aMinDist[iTrials * nRows + iStartRow];
            const algorithmFPType * const pAddedCenter = &pLastAddedCenter[iTrials * dim];

            minDistAccTrials[iTrials * nBlock + iBlock] =
                updateMinDistForITrials(pDistSq, iTrials, nRowsToProcess, pData, colIdx, rowIdx, pAddedCenter, weights, pDistSqBest);
        }
        minDistAccTrials[iBestTrial * nBlock + iBlock] =
            updateMinDistForITrials(pDistSqBest, iBestTrial, nRowsToProcess, pData, colIdx, rowIdx, pLastAddedCenter, weights, pDistSqBest);

        return _csr->releaseSparseBlock(block);
    }

    // For each data point from the provided data block, calculate squared distance
    // from current trial center to the rows in the block and update min distance
    algorithmFPType updateMinDistForITrials(algorithmFPType * const pDistSq, size_t iTrials, size_t nRowsToProcess,
                                            const algorithmFPType * const pData, const size_t * const colIdx, const size_t * const rowIdx,
                                            const algorithmFPType * const pLastAddedCenter, const algorithmFPType * const aWeights,
                                            const algorithmFPType * const pDistSqBest)
    {
        algorithmFPType sumOfDist2            = algorithmFPType(0);
        size_t csrCursor                      = 0u;
        algorithmFPType pLastAddedCenterSumSq = algorithmFPType(0.);
        // Calculate sum of squares of the last added center
        for (size_t iCol = 0u; iCol < dim; iCol++)
        {
            pLastAddedCenterSumSq += pLastAddedCenter[iCol] * pLastAddedCenter[iCol];
        }

        for (size_t iRow = 0u; iRow < nRowsToProcess; iRow++)
        {
            algorithmFPType dist2 = pLastAddedCenterSumSq;
            const size_t nValues  = rowIdx[iRow + 1] - rowIdx[iRow];

            // Add sum of squares of the current row to the sum of squares of the last added center
            // Subtract 2 * product of non-zero element of current row and the element at the same index in the lastAddedCenter
            // This gives squared distance between last added center and current row using x^2 + y^2 - 2xy
            for (size_t i = 0u; i < nValues; i++, csrCursor++)
            {
                dist2 += pData[csrCursor] * pData[csrCursor] - 2 * pData[csrCursor] * pLastAddedCenter[colIdx[csrCursor] - 1];
            }
            if (aWeights)
            {
                dist2 *= aWeights[iRow];
            }

            pDistSq[iRow] = daal::services::internal::min<cpu, algorithmFPType>(pDistSqBest[iRow], dist2);
            sumOfDist2 += pDistSq[iRow];
        }

        return sumOfDist2;
    }

    //copy one row from the given table to the destination buffer and return the sum of squares
    //of the data in this row
    algorithmFPType copyOneRowCalcSumSq(size_t iRow, algorithmFPType * pDst) const
    {
        // TODO: Better to use ReadRowsCSR, but there is a bug related to static library linking.
        // Fixme when ReadRowsCSR will be fixed.
        daal::data_management::CSRBlockDescriptor<algorithmFPType> block;
        _csr->getSparseBlock(iRow, 1, daal::data_management::readOnly, block);
        const auto pData  = block.getBlockValuesPtr();
        const auto colIdx = block.getBlockColumnIndicesPtr();
        const auto rowIdx = block.getBlockRowIndicesPtr();

        daal::services::internal::service_memset<algorithmFPType, cpu>(pDst, algorithmFPType(0.), dim);
        algorithmFPType res(0.);
        const size_t nValues = rowIdx[1] - rowIdx[0];
        for (size_t i = 0; i < nValues; ++i)
        {
            const auto val = pData[i];
            res += val * val;
            const auto colIndex = colIdx[i];
            pDst[colIndex - 1]  = val;
        }
        _csr->releaseSparseBlock(block);
        return res;
    }

public:
    const size_t dim;
    const size_t nRows;

protected:
    NumericTable * _nt;
    CSRNumericTable * _csr;
};

//Base task class for kmeans++ and kmeans||
template <typename algorithmFPType, CpuType cpu, typename DataHelper>
class TaskPlusPlusBatchBase
{
public:
    TaskPlusPlusBatchBase(NumericTable * ntData, NumericTable * ntClusters, size_t numClusters, size_t nTrials, engines::BatchBase & engine)
        : _data(ntData), _ntClusters(ntClusters), _nClusters(numClusters), _nTrials(nTrials), _trialBest(0u), _engine(engine)

    {
        _aMinDist.reset(_data.nRows * _nTrials);
        _overallError.reset(_nTrials);

        _nBlocks = _data.nRows / _nRowsInBlock;
        _nBlocks += (_nBlocks * _nRowsInBlock != _data.nRows);

        _aMinDistAcc.reset(_nBlocks * _nTrials);
    }

protected:
    //copy _dim*nPt algorithmFPType values from pSrc to pDst
    services::Status copyPoints(algorithmFPType * pDst, const algorithmFPType * pSrc, size_t nPt) const
    {
        int result =
            daal::services::internal::daal_memcpy_s(pDst, sizeof(algorithmFPType) * _data.dim * nPt, pSrc, sizeof(algorithmFPType) * _data.dim * nPt);
        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }
    //get first center at random
    size_t calcFirstCenter();

    //find a row corresponding to the sample
    size_t findSample(algorithmFPType sample);

    //update minimal distance using last added center
    Status updateMinDist(const algorithmFPType * aWeights, size_t nTrials);

    //current value of overall error (goal function)
    algorithmFPType overallError() const { return _overallError[_trialBest]; }

    //generate _aProbability array data
    Status generateProbabilities(size_t iStart, size_t nTotal)
    {
        return UniformKernelDefault<algorithmFPType, cpu>::compute(algorithmFPType(0.), algorithmFPType(1.), _engine, nTotal,
                                                                   _aProbability.get() + iStart);
    }

protected:
    DataHelper _data;
    NumericTable * _ntClusters;
    const size_t _nClusters;
    size_t _nTrials;
    size_t _trialBest;
    engines::BatchBase & _engine;
    size_t _nBlocks;

    TArray<algorithmFPType, cpu> _lastAddedCenter; //center last added to the clusters for all trials (nTrials x nDims)
    algorithmFPType _lastAddedCenterSumSq;         //sum of squares of last added center
    TArray<algorithmFPType, cpu> _aMinDist;        //distance to the nearest cluster for every point for all trials (nTrials x nRows)
    TArray<algorithmFPType, cpu> _aMinDistAcc;     //accumulated aMinDist per every block for all trials (nTrials x nBlock)
    TArray<algorithmFPType, cpu> _overallError;    //current value of overall error (goal function) for all trials (nTrials x 1)
    TArray<algorithmFPType, cpu> _aProbability;    //array of probabilities for all trials (nTrials x nCluster)
};

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
class TaskPlusPlusBatch : public TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper>
{
public:
    typedef TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper> super;
    TaskPlusPlusBatch(NumericTable * ntData, const algorithmFPType * aWeight, NumericTable * ntClusters, size_t numClusters, size_t nTrials,
                      engines::BatchBase & engine)
        : super(ntData, ntClusters, numClusters, nTrials, engine), _aWeight(aWeight)
    {
        this->_lastAddedCenterSumSq = algorithmFPType(0);
        this->_lastAddedCenter.reset(this->_data.dim * this->_nTrials); //reserve memory for a single point only
        this->_aProbability.reset(numClusters * this->_nTrials);        //reserve memory for all candidates
    }
    Status run();

protected:
    void calcCenter(size_t iCluster);
    size_t samplePoint(size_t iCluster);

protected:
    const algorithmFPType * _aWeight;
};

template <typename algorithmFPType, CpuType cpu>
struct TlsPPData
{
    algorithmFPType * gemmResult; //result of gemm call is placed here
    algorithmFPType accMinDist2;  //goal function accumulated for all blocks processed by a thread
    int aCandidateRating[1];      //rating of candidates updated in all blocks processed by a thread
};

//TaskParallelPlusUpdateDist class is used by kmeans init batch and distributed kernel
template <typename algorithmFPType, CpuType cpu, typename DataHelper>
class TaskParallelPlusUpdateDist
{
protected:
    typedef TlsPPData<algorithmFPType, cpu> TlsPPData_t;

public:
    TaskParallelPlusUpdateDist(size_t nBlocks, int * aCandidateRating, int * aNearestCandidateIdx, algorithmFPType & overallError, DataHelper & data,
                               const algorithmFPType * lastAddedCenters, algorithmFPType * lastAddedCenterNorm2, algorithmFPType * aMinDist,
                               algorithmFPType * aMinDistAcc)
        : _nBlocks(nBlocks),
          _aCandidateRating(aCandidateRating),
          _aNearestCandidateIdx(aNearestCandidateIdx),
          _overallError(overallError),
          _data(data),
          _lastAddedCenter(lastAddedCenters),
          _lastAddedCenterNorm2(lastAddedCenterNorm2),
          _aMinDist(aMinDist),
          _aMinDistAcc(aMinDistAcc)
    {}
    Status updateMinDist(size_t iFirstOfNewCandidates, size_t nNewCandidates);

protected:
    Status processBlock(size_t iBlock, TlsPPData_t * tlsLocal, size_t iFirstOfNewCandidates, size_t nNewCandidates);
    bool findBestCandidate(typename DataHelper::BlockHelperType & blockHelper, size_t iRow, algorithmFPType * pDistSq, size_t nRowsToProcess,
                           size_t nNewCandidates, size_t & iBestCandidate, const algorithmFPType * gemmResult) const;

protected:
    size_t _nBlocks;
    int * _aCandidateRating;
    int * _aNearestCandidateIdx;
    algorithmFPType & _overallError;
    DataHelper & _data;
    const algorithmFPType * _lastAddedCenter;
    algorithmFPType * _lastAddedCenterNorm2;
    algorithmFPType * _aMinDist;
    algorithmFPType * _aMinDistAcc;
};

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
class TaskParallelPlusBatch : public TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper>
{
public:
    typedef TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper> super;
    TaskParallelPlusBatch(NumericTable * ntData, NumericTable * ntClusters, const Parameter & par, engines::BatchBase & engine)
        : super(ntData, ntClusters, par.nClusters, 1, engine),
          _L(par.oversamplingFactor * par.nClusters),
          _R(par.nRounds),
          _nCandidates(0),
          _aNearestCandidateIdx(this->_data.nRows)
    {
        this->_lastAddedCenter.reset(this->_data.dim * _L); //reserve memory for L points
        _lastAddedCenterNorm2.reset(_L);                    //reserve memory for L values
        _aNearestCandidateIdx.reset(this->_data.nRows);
    }
    Status run();

private:
    typedef services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > HomogenNumericTableCPUPtr;

private:
    Status updateMinDist(size_t iFirstOfNewCandidates, size_t nNewCandidates);
    size_t calcCenters(size_t nRequired, size_t * aCenters, size_t iRound);
    //sample nPt points with probability proportional to their contribution to the overall error,
    //put result to aPt
    size_t samplePoints(size_t nPt, size_t * aPt, size_t iRound);
    Status getCandidates(HomogenNumericTableCPUPtr & pCandidates);

private:
    const size_t _L;
    const size_t _R;
    //Note: maxNumberOfCandidates = _L*_R + 1;
    size_t _nCandidates;                                //number of candidates found so far (from 0 to maxNumberOfCandidates)
    TArray<size_t, cpu> _aCandidateIdx;                 //array[maxNumberOfCandidates], contains row indices of the candidates added so far
    TArray<int, cpu> _aCandidateRating;                 //array[maxNumberOfCandidates], number of points closest to each candidate found so far
    TArray<int, cpu> _aNearestCandidateIdx;             //index of the nearest candidate in _aCandidateIdx per each point
    TArray<algorithmFPType, cpu> _lastAddedCenterNorm2; //array[L] contains 0.5*(center, center) for each last added center
};

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskPlusPlusBatch<algorithmFPType, cpu, DataHelper>::run()
{
    services::Status status;

    DAAL_CHECK(this->_aMinDist.get() && this->_aMinDistAcc.get() && this->_lastAddedCenter.get() && this->_aProbability.get(),
               ErrorMemoryAllocationFailed);
    WriteOnlyRows<algorithmFPType, cpu> clustersBD(this->_ntClusters, 0u, this->_nClusters);
    DAAL_CHECK_BLOCK_STATUS(clustersBD);
    algorithmFPType * const clusters = clustersBD.get();

    daal::services::internal::service_memset<algorithmFPType, cpu>(this->_aMinDist.get(), daal::services::internal::MaxVal<algorithmFPType>::get(),
                                                                   this->_data.nRows * this->_nTrials);

    this->generateProbabilities(0u, this->_nClusters * this->_nTrials);

    //get first center at random
    this->calcFirstCenter();

    //copy it to the result
    status |= this->copyPoints(&clusters[0u * this->_data.dim], &this->_lastAddedCenter[0u * this->_data.dim], 1u);

    // for first centroids is one trial
    this->updateMinDist(_aWeight, 1u);

    //get other centers
    for (size_t iCluster = 1u; iCluster < this->_nClusters; iCluster++)
    {
        calcCenter(iCluster);
        //copy it to the result
        status |= this->copyPoints(&clusters[iCluster * this->_data.dim], &this->_lastAddedCenter[this->_trialBest * this->_data.dim], 1u);
    }

    return status;
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
size_t TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper>::calcFirstCenter()
{
    //use first element of probabilities array to sample a new center
    const algorithmFPType prob = this->_aProbability[0];
    size_t iRow                = prob * _data.nRows;
    if (iRow == _data.nRows) //round-off error
    {
        --iRow;
    }
    _lastAddedCenterSumSq = this->_data.copyOneRowCalcSumSq(iRow, &_lastAddedCenter[0 * this->_data.dim]);
    return iRow;
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
size_t TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper>::findSample(algorithmFPType sample)
{
    const algorithmFPType * const aMinDistAcc = &_aMinDistAcc[_trialBest * _nBlocks];
    const algorithmFPType * const aMinDist    = &_aMinDist[_trialBest * _data.nRows];
    //find the block this sample belongs to
    size_t iBlock = 0;
    for (; (iBlock + 1 < _nBlocks) && (sample >= aMinDistAcc[iBlock]); ++iBlock)
    {
        sample -= aMinDistAcc[iBlock];
    }

    //find the row in the block corresponding to the sample
    size_t nRowsToProcess = _nRowsInBlock;
    if (iBlock == _nBlocks - 1)
    {
        nRowsToProcess = _data.nRows - iBlock * _nRowsInBlock;
    }
    const size_t iStartRow = iBlock * _nRowsInBlock;
    size_t iRow            = 0;
    for (; (iRow + 1 < nRowsToProcess) && (sample >= aMinDist[iStartRow + iRow]); ++iRow)
    {
        sample -= aMinDist[iStartRow + iRow];
    }
    return iStartRow + iRow;
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
size_t TaskPlusPlusBatch<algorithmFPType, cpu, DataHelper>::samplePoint(size_t iCluster)
{
    const algorithmFPType eps  = algorithmFPType(0.1) * this->overallError() / algorithmFPType(this->_data.nRows);
    algorithmFPType * aMinDist = this->_aMinDist.get();

    //take a pre-computed probability value
    algorithmFPType probability = this->_aProbability.get()[iCluster];
    do
    {
        size_t iRow = this->findSample(this->overallError() * probability);
        if (aMinDist[iRow] > eps)
        {
            aMinDist[iRow] = 0;
            return iRow;
        }
        //already taken or duplicate point, sample again
        UniformKernelDefault<algorithmFPType, cpu>::compute(algorithmFPType(0.), algorithmFPType(1.), this->_engine, 1, &probability);
    } while (true);
    return 0;
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper>::updateMinDist(const algorithmFPType * aWeights, size_t nTrials)
{
    SafeStatus safeStat;
    daal::threader_for(_nBlocks, _nBlocks, [=, &safeStat](size_t iBlock) {
        safeStat |=
            _data.updateMinDistInBlock(_aMinDistAcc.get(), _nBlocks, iBlock, nTrials, _trialBest, aWeights, _lastAddedCenter.get(), _aMinDist.get());
    });

    DAAL_CHECK_SAFE_STATUS();

    for (size_t iTrials = 0u; iTrials < nTrials; iTrials++)
    {
        algorithmFPType distance = _aMinDistAcc[iTrials * _nBlocks + 0u];
        for (size_t iBlock = 1u; iBlock < _nBlocks; iBlock++)
        {
            distance += _aMinDistAcc[iTrials * _nBlocks + iBlock];
        }
        _overallError[iTrials] = distance;
    }
    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
void TaskPlusPlusBatch<algorithmFPType, cpu, DataHelper>::calcCenter(size_t iCluster)
{
    // nTrials new candidats
    for (size_t iTrials = 0u; iTrials < this->_nTrials; iTrials++)
    {
        const algorithmFPType probability = this->_aProbability[iTrials * this->_nClusters + iCluster];

        const size_t iRow           = this->_nTrials == 1 ? this->samplePoint(iCluster) : this->findSample(this->overallError() * probability);
        this->_lastAddedCenterSumSq = this->_data.copyOneRowCalcSumSq(iRow, &this->_lastAddedCenter[iTrials * this->_data.dim]);
    }

    // for one trial, there is no need to recalculate the inertia on the last selected cluster
    if (this->_nTrials == 1 && iCluster == this->_nClusters - 1)
    {
        return;
    }

    this->updateMinDist(_aWeight, this->_nTrials);

    // search best candidate from nTrials
    algorithmFPType bestMinInertia = daal::services::internal::MaxVal<algorithmFPType>::get();
    size_t iTialBest               = 0u;

    for (size_t iTrials = 0u; iTrials < this->_nTrials; iTrials++)
    {
        algorithmFPType newInersia = this->_overallError[iTrials];

        if (newInersia < bestMinInertia)
        {
            bestMinInertia = newInersia;
            iTialBest      = iTrials;
        }
    }

    this->_trialBest = iTialBest;
}

template <typename algorithmFPType, CpuType cpu>
services::Status KMeansInitKernel<plusPlusDense, algorithmFPType, cpu>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                                const NumericTable * const * r, const Parameter * par,
                                                                                engines::BatchBase & engine)
{
    TaskPlusPlusBatch<algorithmFPType, cpu, DataHelperDense<algorithmFPType, cpu> > task(const_cast<NumericTable *>(a[0]), //data
                                                                                         nullptr,
                                                                                         const_cast<NumericTable *>(r[0]), //clusters
                                                                                         par->nClusters, par->nTrials, engine);

    return task.run();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KMeansInitKernel<plusPlusCSR, algorithmFPType, cpu>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                              const NumericTable * const * r, const Parameter * par,
                                                                              engines::BatchBase & engine)
{
    TaskPlusPlusBatch<algorithmFPType, cpu, DataHelperCSR<algorithmFPType, cpu> > task(const_cast<NumericTable *>(a[0]), //data
                                                                                       nullptr,
                                                                                       const_cast<NumericTable *>(r[0]), //clusters
                                                                                       par->nClusters, par->nTrials, engine);
    return task.run();
}

//////////////////////// TaskParallelPlusBatch ///////////////////////////////////
template <typename algorithmFPType, CpuType cpu, typename DataHelper>
size_t TaskParallelPlusBatch<algorithmFPType, cpu, DataHelper>::samplePoints(size_t nPt, size_t * aPt, size_t iRound)
{
    if (iRound >= _R)
    {
        //generate extra values in _aProbability
        this->generateProbabilities(_nCandidates, nPt);
    }
    //sample each point independently
    daal::threader_for(nPt, nPt, [=](size_t iPt) {
        const size_t iCandidate     = _nCandidates + iPt;
        algorithmFPType probability = this->_aProbability.get()[iCandidate];
        aPt[iPt]                    = this->findSample(this->overallError() * probability);
    });

    const algorithmFPType eps = algorithmFPType(0.1) * this->overallError() / algorithmFPType(this->_data.nRows);
    size_t iNewCandidate      = 0;
    //update ratings and check for the duplicates
    algorithmFPType * aMinDist = this->_aMinDist.get();
    for (size_t iPt = 0; iPt < nPt; ++iPt)
    {
        const size_t iRow = aPt[iPt];
        if (aMinDist[iRow] > eps)
        {
            aMinDist[iRow]            = 0;
            auto pNearestCandIndex    = _aNearestCandidateIdx.get();
            const auto iPrevCandidate = pNearestCandIndex[iRow];
            pNearestCandIndex[iRow] =
                _nCandidates + iNewCandidate; //this point becomes an (_nCandidates + iNewCandidate)-th candidate, increases own rating
            _aCandidateRating.get()[iPrevCandidate] -= 1;
            _aCandidateRating.get()[_nCandidates + iNewCandidate] += 1;
            aPt[iNewCandidate] = iRow;
            ++iNewCandidate;
        }
    }
    return iNewCandidate;
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
bool TaskParallelPlusUpdateDist<algorithmFPType, cpu, DataHelper>::findBestCandidate(typename DataHelper::BlockHelperType & blockHelper, size_t iRow,
                                                                                     algorithmFPType * pDistSq, size_t nRowsToProcess,
                                                                                     size_t nNewCandidates, size_t & iBestCandidate,
                                                                                     const algorithmFPType * gemmResult) const
{
    size_t iCandidate = 0;
    iBestCandidate    = iCandidate;
    algorithmFPType valBest =
        _lastAddedCenterNorm2[iCandidate] - blockHelper.getGemmResult(iRow, iCandidate, nRowsToProcess, nNewCandidates, gemmResult);
    for (iCandidate = 1; iCandidate < nNewCandidates; ++iCandidate)
    {
        algorithmFPType valCand =
            _lastAddedCenterNorm2[iCandidate] - blockHelper.getGemmResult(iRow, iCandidate, nRowsToProcess, nNewCandidates, gemmResult);
        if (valBest > valCand)
        {
            valBest        = valCand;
            iBestCandidate = iCandidate;
        }
    }
    const algorithmFPType * pLastAddedCenter = _lastAddedCenter;
    const algorithmFPType dist2              = blockHelper.getRowSumSq(iRow, pLastAddedCenter + iBestCandidate * this->_data.dim);
    if (dist2 < pDistSq[iRow])
    {
        pDistSq[iRow] = dist2;
        return true;
    }
    return false;
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskParallelPlusUpdateDist<algorithmFPType, cpu, DataHelper>::processBlock(size_t iBlock, TlsPPData_t * tlsLocal, size_t iFirstOfNewCandidates,
                                                                                  size_t nNewCandidates)
{
    int * aCandidateRating = tlsLocal->aCandidateRating;
    size_t nRowsToProcess  = _nRowsInBlock;
    if (iBlock == _nBlocks - 1) nRowsToProcess = _data.nRows - iBlock * _nRowsInBlock;
    const size_t iStartRow = iBlock * _nRowsInBlock;

    typename DataHelper::BlockHelperType blockHelper(_data.ntIface(), _data.dim, iStartRow, nRowsToProcess);
    DAAL_CHECK_BLOCK_STATUS(blockHelper);
    blockHelper.callGemm(_lastAddedCenter, nRowsToProcess, nNewCandidates, tlsLocal->gemmResult);

    algorithmFPType * pDistSq  = _aMinDist + iStartRow;
    auto * pNearestCandIndex   = _aNearestCandidateIdx + iStartRow;
    algorithmFPType sumOfDist2 = 0;
    for (size_t iRow = 0; iRow < nRowsToProcess; ++iRow)
    {
        size_t iBestCandidate = 0;
        if (findBestCandidate(blockHelper, iRow, pDistSq, nRowsToProcess, nNewCandidates, iBestCandidate, tlsLocal->gemmResult))
        {
            const auto iPrevCandidate = pNearestCandIndex[iRow];
            pNearestCandIndex[iRow]   = iFirstOfNewCandidates + iBestCandidate;
            aCandidateRating[iPrevCandidate] -= 1;
            aCandidateRating[iFirstOfNewCandidates + iBestCandidate] += 1;
        }
        sumOfDist2 += pDistSq[iRow];
    }
    _aMinDistAcc[iBlock] = sumOfDist2;
    tlsLocal->accMinDist2 += sumOfDist2;
    return Status();
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskParallelPlusUpdateDist<algorithmFPType, cpu, DataHelper>::updateMinDist(size_t iFirstOfNewCandidates, size_t nNewCandidates)
{
    const size_t nCandidates  = iFirstOfNewCandidates + nNewCandidates;
    const size_t gemmDataSize = _nRowsInBlock * nCandidates;
    daal::static_tls<TlsPPData_t *> tlsData([=]() -> TlsPPData_t * {
        const size_t sz     = sizeof(TlsPPData_t) + (nCandidates - 1) * sizeof(int);
        byte * ptr          = service_scalable_calloc<byte, cpu>(sz);
        TlsPPData_t * pData = new (ptr) TlsPPData_t;
        //allocate memory for Intel(R) MKL result
        if (pData)
        {
            pData->gemmResult = service_calloc<algorithmFPType, cpu>(gemmDataSize);
            if (!pData->gemmResult)
            {
                service_scalable_free<byte, cpu>(ptr);
                return nullptr;
            }
        }
        return pData;
    });
    bool bMemoryAllocationFailed = false;
    algorithmFPType newOverallError(0.);
    SafeStatus safeStat;
    daal::static_threader_for(this->_nBlocks, [=, &tlsData, &bMemoryAllocationFailed, &safeStat](size_t iBlock, size_t tid) {
        TlsPPData_t * tlsLocal = tlsData.local(tid);
        if (!tlsLocal)
        {
            bMemoryAllocationFailed = true;
            return;
        }
        safeStat |= processBlock(iBlock, tlsLocal, iFirstOfNewCandidates, nNewCandidates);
    });
    tlsData.reduce([=, &newOverallError](TlsPPData_t * ptr) -> void {
        if (!ptr) return;
        newOverallError += ptr->accMinDist2;
        for (size_t j = 0; j < nCandidates; ++j) _aCandidateRating[j] += ptr->aCandidateRating[j];
        service_free<algorithmFPType, cpu>(ptr->gemmResult);
        service_scalable_free<byte, cpu>((byte *)ptr);
    });
    this->_overallError = newOverallError;
    if (!safeStat) return safeStat.detach();
    return bMemoryAllocationFailed ? Status(ErrorMemoryAllocationFailed) : Status();
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskParallelPlusBatch<algorithmFPType, cpu, DataHelper>::updateMinDist(size_t iFirstOfNewCandidates, size_t nNewCandidates)
{
    TaskParallelPlusUpdateDist<algorithmFPType, cpu, DataHelper> impl(
        this->_nBlocks, _aCandidateRating.get(), _aNearestCandidateIdx.get(), this->_overallError[this->_trialBest], this->_data,
        this->_lastAddedCenter.get(), this->_lastAddedCenterNorm2.get(), this->_aMinDist.get(), this->_aMinDistAcc.get());
    return impl.updateMinDist(iFirstOfNewCandidates, nNewCandidates);
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskParallelPlusBatch<algorithmFPType, cpu, DataHelper>::getCandidates(HomogenNumericTableCPUPtr & pCandidates)
{
    const size_t maxNumberOfCandidates = _L * _R + 1;
    //reserve memory and init work variables for all candidates
    _aCandidateIdx.reset(maxNumberOfCandidates);
    _aCandidateRating.reset(maxNumberOfCandidates);
    this->_aProbability.reset(maxNumberOfCandidates);
    _aNearestCandidateIdx.reset(this->_data.nRows);
    DAAL_CHECK(_aCandidateIdx.get() && _aCandidateRating.get() && _aNearestCandidateIdx.get() && this->_aProbability.get(),
               ErrorMemoryAllocationFailed);

    this->generateProbabilities(0, maxNumberOfCandidates);

    //get first candidate at random
    auto iCenter                         = this->calcFirstCenter();
    this->_lastAddedCenterNorm2.get()[0] = algorithmFPType(0.5) * this->_lastAddedCenterSumSq;
    super::updateMinDist(nullptr, 1);

    Status s;
    //create data for candidates output
    pCandidates = HomogenNumericTableCPU<algorithmFPType, cpu>::create(this->_data.dim, maxNumberOfCandidates, &s);
    DAAL_CHECK_STATUS_VAR(s);
    {
        WriteOnlyRows<algorithmFPType, cpu> candidatesBD(pCandidates.get(), 0, pCandidates->getNumberOfRows());
        DAAL_CHECK_BLOCK_STATUS(candidatesBD);

        //no ratings yet
        daal::services::internal::service_memset<int, cpu>(_aCandidateRating.get(), 0, maxNumberOfCandidates);

        //add first candidate
        _aCandidateIdx.get()[0] = iCenter;
        _nCandidates            = 1;
        s |= this->copyPoints(candidatesBD.get(), this->_lastAddedCenter.get(), 1);
        //it is nearest for all points
        daal::services::internal::service_memset<int, cpu>(_aNearestCandidateIdx.get(), 0, this->_data.nRows);
        //hence its rating is highest so far
        _aCandidateRating.get()[0] = this->_data.nRows;

        //get other candidates in R rounds
        bool bDone = false;
        for (size_t iRound = 0; !bDone; ++iRound)
        {
            //calculate candidates: _L or min(_L, what remains);
            const size_t nRequired      = (iRound < _R || _L < (maxNumberOfCandidates - _nCandidates)) ? _L : (maxNumberOfCandidates - _nCandidates);
            const size_t nNewCandidates = calcCenters(nRequired, _aCandidateIdx.get() + _nCandidates, iRound);
            if (nNewCandidates)
            {
                //copy them to the candidates table
                s |= this->copyPoints(candidatesBD.get() + _nCandidates * this->_data.dim, this->_lastAddedCenter.get(), nNewCandidates);
                const size_t iFirstNewCandidate = _nCandidates;
                _nCandidates += nNewCandidates;
                DAAL_CHECK_STATUS(s, this->updateMinDist(iFirstNewCandidate, nNewCandidates));
            }
            bDone = ((iRound + 1 >= _R) && (_nCandidates > this->_nClusters));
        }
    }
    if (_nCandidates < maxNumberOfCandidates) pCandidates->resize(_nCandidates);
    return s;
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskParallelPlusBatch<algorithmFPType, cpu, DataHelper>::run()
{
    DAAL_CHECK(this->_aMinDist.get() && this->_aMinDistAcc.get() && this->_lastAddedCenter.get() && _lastAddedCenterNorm2.get(),
               ErrorMemoryAllocationFailed);
    daal::services::internal::service_memset<algorithmFPType, cpu>(this->_aMinDist.get(), daal::services::internal::MaxVal<algorithmFPType>::get(),
                                                                   this->_data.nRows);
    HomogenNumericTableCPUPtr pCandidates;
    Status s = getCandidates(pCandidates);
    if (!s) return s;
    const auto nCandidates = pCandidates->getNumberOfRows();

    TArray<algorithmFPType, cpu> aWeight(nCandidates);
    const algorithmFPType div(1. / algorithmFPType(this->_data.nRows));
    for (auto i = 0; i < nCandidates; ++i) aWeight.get()[i] = div * algorithmFPType(_aCandidateRating.get()[i]);
    TaskPlusPlusBatch<algorithmFPType, cpu, DataHelperDense<algorithmFPType, cpu> > task(pCandidates.get(), aWeight.get(), this->_ntClusters,
                                                                                         this->_nClusters, 1, this->_engine);
    return task.run();
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
size_t TaskParallelPlusBatch<algorithmFPType, cpu, DataHelper>::calcCenters(size_t nRequired, size_t * aCenters, size_t iRound)
{
    const size_t nPt = samplePoints(nRequired, aCenters, iRound);
    //copy points in parallel
    daal::threader_for(nPt, nPt, [=](size_t iPt) {
        const size_t iRow = aCenters[iPt];
        this->_lastAddedCenterNorm2.get()[iPt] =
            algorithmFPType(0.5) * this->_data.copyOneRowCalcSumSq(iRow, this->_lastAddedCenter.get() + iPt * this->_data.dim);
    });
    return nPt;
}

template <typename algorithmFPType, CpuType cpu>
services::Status KMeansInitKernel<parallelPlusDense, algorithmFPType, cpu>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                                    const NumericTable * const * r, const Parameter * par,
                                                                                    engines::BatchBase & engine)
{
    TaskParallelPlusBatch<algorithmFPType, cpu, DataHelperDense<algorithmFPType, cpu> > task(const_cast<NumericTable *>(a[0]), //data
                                                                                             const_cast<NumericTable *>(r[0]), //clusters
                                                                                             *par, engine);
    return task.run();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KMeansInitKernel<parallelPlusCSR, algorithmFPType, cpu>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                                  const NumericTable * const * r, const Parameter * par,
                                                                                  engines::BatchBase & engine)
{
    TaskParallelPlusBatch<algorithmFPType, cpu, DataHelperCSR<algorithmFPType, cpu> > task(const_cast<NumericTable *>(a[0]), //data
                                                                                           const_cast<NumericTable *>(r[0]), //clusters
                                                                                           *par, engine);
    return task.run();
}

} // namespace internal
} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal
