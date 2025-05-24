/* file: daal_strings.h */
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
//  String variables.
//--
*/

#ifndef __DAAL_STRINGS_H__
#define __DAAL_STRINGS_H__

/** \file daal_strings.h */

#include "services/daal_defines.h"

/**
 * @ingroup error_handling
 * @{
 */
#define DAAL_STRINGS_LIST()                                      \
    DECLARE_DAAL_STRING_CONST(basicStatisticsSum)                \
    DECLARE_DAAL_STRING_CONST(basicStatisticsMinimum)            \
    DECLARE_DAAL_STRING_CONST(basicStatisticsMaximum)            \
    DECLARE_DAAL_STRING_CONST(sortedData)                        \
    DECLARE_DAAL_STRING_CONST(normalizedData)                    \
    DECLARE_DAAL_STRING_CONST(inputGradient)                     \
    DECLARE_DAAL_STRING_CONST(gradient)                          \
    DECLARE_DAAL_STRING_CONST(gradientSquareSum)                 \
    DECLARE_DAAL_STRING_CONST(value)                             \
    DECLARE_DAAL_STRING_CONST(data)                              \
    DECLARE_DAAL_STRING_CONST(weights)                           \
    DECLARE_DAAL_STRING_CONST(biases)                            \
    DECLARE_DAAL_STRING_CONST(populationMean)                    \
    DECLARE_DAAL_STRING_CONST(populationVariance)                \
    DECLARE_DAAL_STRING_CONST(resultLayerData)                   \
    DECLARE_DAAL_STRING_CONST(auxData)                           \
    DECLARE_DAAL_STRING_CONST(auxWeights)                        \
    DECLARE_DAAL_STRING_CONST(auxMean)                           \
    DECLARE_DAAL_STRING_CONST(auxStandardDeviation)              \
    DECLARE_DAAL_STRING_CONST(auxSelectedIndices)                \
    DECLARE_DAAL_STRING_CONST(auxPopulationMean)                 \
    DECLARE_DAAL_STRING_CONST(auxPopulationVariance)             \
    DECLARE_DAAL_STRING_CONST(auxProbabilities)                  \
    DECLARE_DAAL_STRING_CONST(auxGroundTruth)                    \
    DECLARE_DAAL_STRING_CONST(inputLayerData)                    \
    DECLARE_DAAL_STRING_CONST(auxMask)                           \
    DECLARE_DAAL_STRING_CONST(impurityThreshold)                 \
    DECLARE_DAAL_STRING_CONST(inputGradientCollection)           \
    DECLARE_DAAL_STRING_CONST(valueCollection)                   \
    DECLARE_DAAL_STRING_CONST(biasDerivatives)                   \
    DECLARE_DAAL_STRING_CONST(weightDerivatives)                 \
    DECLARE_DAAL_STRING_CONST(weightsDimension)                  \
    DECLARE_DAAL_STRING_CONST(dataDimension)                     \
    DECLARE_DAAL_STRING_CONST(correlationDistance)               \
    DECLARE_DAAL_STRING_CONST(cosineDistance)                    \
    DECLARE_DAAL_STRING_CONST(quantiles)                         \
    DECLARE_DAAL_STRING_CONST(quantileOrders)                    \
    DECLARE_DAAL_STRING_CONST(covariance)                        \
    DECLARE_DAAL_STRING_CONST(correlation)                       \
    DECLARE_DAAL_STRING_CONST(mean)                              \
    DECLARE_DAAL_STRING_CONST(minObservationsInLeafNode)         \
    DECLARE_DAAL_STRING_CONST(minObservationsInSplitNode)        \
    DECLARE_DAAL_STRING_CONST(minWeightFractionInLeafNode)       \
    DECLARE_DAAL_STRING_CONST(minImpurityDecreaseInSplitNode)    \
    DECLARE_DAAL_STRING_CONST(maxLeafNodes)                      \
    DECLARE_DAAL_STRING_CONST(sum)                               \
    DECLARE_DAAL_STRING_CONST(penaltyL1)                         \
    DECLARE_DAAL_STRING_CONST(penaltyL2)                         \
    DECLARE_DAAL_STRING_CONST(permutationMatrix)                 \
    DECLARE_DAAL_STRING_CONST(permutedColumns)                   \
    DECLARE_DAAL_STRING_CONST(matrixQ)                           \
    DECLARE_DAAL_STRING_CONST(matrixR)                           \
    DECLARE_DAAL_STRING_CONST(eigenvalue)                        \
    DECLARE_DAAL_STRING_CONST(eigenvalues)                       \
    DECLARE_DAAL_STRING_CONST(eigenvectors)                      \
    DECLARE_DAAL_STRING_CONST(nObservationsCorrelation)          \
    DECLARE_DAAL_STRING_CONST(crossProductCorrelation)           \
    DECLARE_DAAL_STRING_CONST(crossProduct)                      \
    DECLARE_DAAL_STRING_CONST(nObservationsSVD)                  \
    DECLARE_DAAL_STRING_CONST(sumSquaresSVD)                     \
    DECLARE_DAAL_STRING_CONST(sumSVD)                            \
    DECLARE_DAAL_STRING_CONST(sumCorrelation)                    \
    DECLARE_DAAL_STRING_CONST(auxiliaryData)                     \
    DECLARE_DAAL_STRING_CONST(nObservations)                     \
    DECLARE_DAAL_STRING_CONST(partialMinimum)                    \
    DECLARE_DAAL_STRING_CONST(partialMaximum)                    \
    DECLARE_DAAL_STRING_CONST(partialSum)                        \
    DECLARE_DAAL_STRING_CONST(partialSumSquares)                 \
    DECLARE_DAAL_STRING_CONST(partialSumSquaresCentered)         \
    DECLARE_DAAL_STRING_CONST(minimum)                           \
    DECLARE_DAAL_STRING_CONST(maximum)                           \
    DECLARE_DAAL_STRING_CONST(sumSquares)                        \
    DECLARE_DAAL_STRING_CONST(sumSquaresCentered)                \
    DECLARE_DAAL_STRING_CONST(secondOrderRawMoment)              \
    DECLARE_DAAL_STRING_CONST(variance)                          \
    DECLARE_DAAL_STRING_CONST(variances)                         \
    DECLARE_DAAL_STRING_CONST(standardDeviation)                 \
    DECLARE_DAAL_STRING_CONST(variation)                         \
    DECLARE_DAAL_STRING_CONST(X)                                 \
    DECLARE_DAAL_STRING_CONST(Y)                                 \
    DECLARE_DAAL_STRING_CONST(values)                            \
    DECLARE_DAAL_STRING_CONST(rowIndexX)                         \
    DECLARE_DAAL_STRING_CONST(rowIndexY)                         \
    DECLARE_DAAL_STRING_CONST(rowIndexResult)                    \
    DECLARE_DAAL_STRING_CONST(minSupport)                        \
    DECLARE_DAAL_STRING_CONST(minSplitLoss)                      \
    DECLARE_DAAL_STRING_CONST(minConfidence)                     \
    DECLARE_DAAL_STRING_CONST(nUniqueItems)                      \
    DECLARE_DAAL_STRING_CONST(nTransactions)                     \
    DECLARE_DAAL_STRING_CONST(maxBins)                           \
    DECLARE_DAAL_STRING_CONST(minBinSize)                        \
    DECLARE_DAAL_STRING_CONST(maxItemsetSize)                    \
    DECLARE_DAAL_STRING_CONST(minItemsetSize)                    \
    DECLARE_DAAL_STRING_CONST(largeItemsets)                     \
    DECLARE_DAAL_STRING_CONST(largeItemsetsSupport)              \
    DECLARE_DAAL_STRING_CONST(antecedentItemsets)                \
    DECLARE_DAAL_STRING_CONST(consequentItemsets)                \
    DECLARE_DAAL_STRING_CONST(confidence)                        \
    DECLARE_DAAL_STRING_CONST(priorClassEstimates)               \
    DECLARE_DAAL_STRING_CONST(alpha)                             \
    DECLARE_DAAL_STRING_CONST(epsilon)                           \
    DECLARE_DAAL_STRING_CONST(weakLearnerTraining)               \
    DECLARE_DAAL_STRING_CONST(weakLearnerPrediction)             \
    DECLARE_DAAL_STRING_CONST(accuracyThreshold)                 \
    DECLARE_DAAL_STRING_CONST(maxIterations)                     \
    DECLARE_DAAL_STRING_CONST(newtonRaphsonAccuracyThreshold)    \
    DECLARE_DAAL_STRING_CONST(newtonRaphsonMaxIterations)        \
    DECLARE_DAAL_STRING_CONST(degenerateCasesThreshold)          \
    DECLARE_DAAL_STRING_CONST(nClasses)                          \
    DECLARE_DAAL_STRING_CONST(nTrees)                            \
    DECLARE_DAAL_STRING_CONST(weightsDegenerateCasesThreshold)   \
    DECLARE_DAAL_STRING_CONST(responsesDegenerateCasesThreshold) \
    DECLARE_DAAL_STRING_CONST(cBound)                            \
    DECLARE_DAAL_STRING_CONST(tau)                               \
    DECLARE_DAAL_STRING_CONST(kernelFunction)                    \
    DECLARE_DAAL_STRING_CONST(training)                          \
    DECLARE_DAAL_STRING_CONST(prediction)                        \
    DECLARE_DAAL_STRING_CONST(labels)                            \
    DECLARE_DAAL_STRING_CONST(predictedLabels)                   \
    DECLARE_DAAL_STRING_CONST(probabilities)                     \
    DECLARE_DAAL_STRING_CONST(groundTruthLabels)                 \
    DECLARE_DAAL_STRING_CONST(groundTruth)                       \
    DECLARE_DAAL_STRING_CONST(supportVectors)                    \
    DECLARE_DAAL_STRING_CONST(classificationCoefficients)        \
    DECLARE_DAAL_STRING_CONST(decisionFunction)                  \
    DECLARE_DAAL_STRING_CONST(beta)                              \
    DECLARE_DAAL_STRING_CONST(confusionMatrix)                   \
    DECLARE_DAAL_STRING_CONST(binaryMetrics)                     \
    DECLARE_DAAL_STRING_CONST(multiClassMetrics)                 \
    DECLARE_DAAL_STRING_CONST(learningRate)                      \
    DECLARE_DAAL_STRING_CONST(logProbabilities)                  \
    DECLARE_DAAL_STRING_CONST(means)                             \
    DECLARE_DAAL_STRING_CONST(goalFunction)                      \
    DECLARE_DAAL_STRING_CONST(objectiveFunction)                 \
    DECLARE_DAAL_STRING_CONST(nIterations)                       \
    DECLARE_DAAL_STRING_CONST(inputWeights)                      \
    DECLARE_DAAL_STRING_CONST(inputCovariances)                  \
    DECLARE_DAAL_STRING_CONST(inputMeans)                        \
    DECLARE_DAAL_STRING_CONST(inputOfStep2)                      \
    DECLARE_DAAL_STRING_CONST(inputOfStep2FromStep1)             \
    DECLARE_DAAL_STRING_CONST(inputOfStep3FromStep1)             \
    DECLARE_DAAL_STRING_CONST(inputOfStep3FromStep2)             \
    DECLARE_DAAL_STRING_CONST(inputOfStep4FromStep3)             \
    DECLARE_DAAL_STRING_CONST(inputOfStep5FromStep2)             \
    DECLARE_DAAL_STRING_CONST(inputOfStep5FromStep3)             \
    DECLARE_DAAL_STRING_CONST(internalInput)                     \
    DECLARE_DAAL_STRING_CONST(internalResult)                    \
    DECLARE_DAAL_STRING_CONST(observationsPerTreeFraction)       \
    DECLARE_DAAL_STRING_CONST(outOfBagError)                     \
    DECLARE_DAAL_STRING_CONST(outOfBagErrorAccuracy)             \
    DECLARE_DAAL_STRING_CONST(outOfBagErrorR2)                   \
    DECLARE_DAAL_STRING_CONST(outOfBagErrorPerObservation)       \
    DECLARE_DAAL_STRING_CONST(outOfBagErrorDecisionFunction)     \
    DECLARE_DAAL_STRING_CONST(outOfBagErrorPrediction)           \
    DECLARE_DAAL_STRING_CONST(outputOfStep1ForStep2)             \
    DECLARE_DAAL_STRING_CONST(outputOfStep1ForStep3)             \
    DECLARE_DAAL_STRING_CONST(outputOfStep2ForStep3)             \
    DECLARE_DAAL_STRING_CONST(outputOfStep4)                     \
    DECLARE_DAAL_STRING_CONST(batchIndices)                      \
    DECLARE_DAAL_STRING_CONST(batchSize)                         \
    DECLARE_DAAL_STRING_CONST(singularValues)                    \
    DECLARE_DAAL_STRING_CONST(rightSingularMatrix)               \
    DECLARE_DAAL_STRING_CONST(leftSingularMatrix)                \
    DECLARE_DAAL_STRING_CONST(SVDNodeCollection)                 \
    DECLARE_DAAL_STRING_CONST(SVDNodeCollectionNT)               \
    DECLARE_DAAL_STRING_CONST(QRNodeCollection)                  \
    DECLARE_DAAL_STRING_CONST(QRNodeCollectionNT)                \
    DECLARE_DAAL_STRING_CONST(qCollection)                       \
    DECLARE_DAAL_STRING_CONST(rCollection)                       \
    DECLARE_DAAL_STRING_CONST(toleranceToConverge)               \
    DECLARE_DAAL_STRING_CONST(indices)                           \
    DECLARE_DAAL_STRING_CONST(auxCenteredData)                   \
    DECLARE_DAAL_STRING_CONST(auxC)                              \
    DECLARE_DAAL_STRING_CONST(auxSigma)                          \
    DECLARE_DAAL_STRING_CONST(dimension)                         \
    DECLARE_DAAL_STRING_CONST(ridgeParameters)                   \
    DECLARE_DAAL_STRING_CONST(nClusters)                         \
    DECLARE_DAAL_STRING_CONST(nRounds)                           \
    DECLARE_DAAL_STRING_CONST(nRowsTotal)                        \
    DECLARE_DAAL_STRING_CONST(offset)                            \
    DECLARE_DAAL_STRING_CONST(oversamplingFactor)                \
    DECLARE_DAAL_STRING_CONST(partialClusters)                   \
    DECLARE_DAAL_STRING_CONST(centroids)                         \
    DECLARE_DAAL_STRING_CONST(inputCentroids)                    \
    DECLARE_DAAL_STRING_CONST(closestClusterDistance)            \
    DECLARE_DAAL_STRING_CONST(closestCluster)                    \
    DECLARE_DAAL_STRING_CONST(numberOfClusters)                  \
    DECLARE_DAAL_STRING_CONST(candidates)                        \
    DECLARE_DAAL_STRING_CONST(candidateRating)                   \
    DECLARE_DAAL_STRING_CONST(partialSums)                       \
    DECLARE_DAAL_STRING_CONST(partialObjectiveFunction)          \
    DECLARE_DAAL_STRING_CONST(partialAssignments)                \
    DECLARE_DAAL_STRING_CONST(partialCandidatesDistances)        \
    DECLARE_DAAL_STRING_CONST(partialCandidatesCentroids)        \
    DECLARE_DAAL_STRING_CONST(assignments)                       \
    DECLARE_DAAL_STRING_CONST(partialClustersNumber)             \
    DECLARE_DAAL_STRING_CONST(gamma)                             \
    DECLARE_DAAL_STRING_CONST(correctionPairs)                   \
    DECLARE_DAAL_STRING_CONST(correctionIndices)                 \
    DECLARE_DAAL_STRING_CONST(averageArgumentLIterations)        \
    DECLARE_DAAL_STRING_CONST(rngState)                          \
    DECLARE_DAAL_STRING_CONST(correctionPairBatchIndices)        \
    DECLARE_DAAL_STRING_CONST(stepLengthSequence)                \
    DECLARE_DAAL_STRING_CONST(nFactors)                          \
    DECLARE_DAAL_STRING_CONST(fullNUsers)                        \
    DECLARE_DAAL_STRING_CONST(usersFactors)                      \
    DECLARE_DAAL_STRING_CONST(factors)                           \
    DECLARE_DAAL_STRING_CONST(featuresPerNode)                   \
    DECLARE_DAAL_STRING_CONST(lambda)                            \
    DECLARE_DAAL_STRING_CONST(preferenceThreshold)               \
    DECLARE_DAAL_STRING_CONST(pyramidHeight)                     \
    DECLARE_DAAL_STRING_CONST(itemsFactors)                      \
    DECLARE_DAAL_STRING_CONST(partialModels)                     \
    DECLARE_DAAL_STRING_CONST(partialModelBlocksToNode)          \
    DECLARE_DAAL_STRING_CONST(partialData)                       \
    DECLARE_DAAL_STRING_CONST(outputOfStep3ForStep4)             \
    DECLARE_DAAL_STRING_CONST(outputOfStep2ForStep4)             \
    DECLARE_DAAL_STRING_CONST(outBlockIndices)                   \
    DECLARE_DAAL_STRING_CONST(blockIndices)                      \
    DECLARE_DAAL_STRING_CONST(usersIndices)                      \
    DECLARE_DAAL_STRING_CONST(itemsIndices)                      \
    DECLARE_DAAL_STRING_CONST(nComponents)                       \
    DECLARE_DAAL_STRING_CONST(auxInputDimensions)                \
    DECLARE_DAAL_STRING_CONST(auxInvMax)                         \
    DECLARE_DAAL_STRING_CONST(nTrials)                           \
    DECLARE_DAAL_STRING_CONST(layer)                             \
    DECLARE_DAAL_STRING_CONST(partialResults)                    \
    DECLARE_DAAL_STRING_CONST(nGroups)                           \
    DECLARE_DAAL_STRING_CONST(kernelSizes)                       \
    DECLARE_DAAL_STRING_CONST(groupDimension)                    \
    DECLARE_DAAL_STRING_CONST(strides)                           \
    DECLARE_DAAL_STRING_CONST(dependentVariable)                 \
    DECLARE_DAAL_STRING_CONST(XTXTable)                          \
    DECLARE_DAAL_STRING_CONST(XTYTable)                          \
    DECLARE_DAAL_STRING_CONST(RTable)                            \
    DECLARE_DAAL_STRING_CONST(QTYTable)                          \
    DECLARE_DAAL_STRING_CONST(interceptFlag)                     \
    DECLARE_DAAL_STRING_CONST(retainRatio)                       \
    DECLARE_DAAL_STRING_CONST(k)                                 \
    DECLARE_DAAL_STRING_CONST(kdTreeTable)                       \
    DECLARE_DAAL_STRING_CONST(distances)                         \
    DECLARE_DAAL_STRING_CONST(auxRetainMask)                     \
    DECLARE_DAAL_STRING_CONST(auxValue)                          \
    DECLARE_DAAL_STRING_CONST(auxSmBeta)                         \
    DECLARE_DAAL_STRING_CONST(argument)                          \
    DECLARE_DAAL_STRING_CONST(valueIdx)                          \
    DECLARE_DAAL_STRING_CONST(variableImportance)                \
    DECLARE_DAAL_STRING_CONST(gradientIdx)                       \
    DECLARE_DAAL_STRING_CONST(hessianIdx)                        \
    DECLARE_DAAL_STRING_CONST(inputArgument)                     \
    DECLARE_DAAL_STRING_CONST(dependentVariables)                \
    DECLARE_DAAL_STRING_CONST(a)                                 \
    DECLARE_DAAL_STRING_CONST(sigma)                             \
    DECLARE_DAAL_STRING_CONST(location)                          \
    DECLARE_DAAL_STRING_CONST(scatter)                           \
    DECLARE_DAAL_STRING_CONST(threshold)                         \
    DECLARE_DAAL_STRING_CONST(conservativeSequence)              \
    DECLARE_DAAL_STRING_CONST(pastUpdateVector)                  \
    DECLARE_DAAL_STRING_CONST(minObservationsInLeafNodes)        \
    DECLARE_DAAL_STRING_CONST(dataForPruning)                    \
    DECLARE_DAAL_STRING_CONST(labelsForPruning)                  \
    DECLARE_DAAL_STRING_CONST(dependentVariablesForPruning)      \
    DECLARE_DAAL_STRING_CONST(partition)                         \
    DECLARE_DAAL_STRING_CONST(transposedData)                    \
    DECLARE_DAAL_STRING_CONST(optimizationSolver)                \
    DECLARE_DAAL_STRING_CONST(outputOfInitForComputeStep3)       \
    DECLARE_DAAL_STRING_CONST(offsets)                           \
    DECLARE_DAAL_STRING_CONST(model)                             \
    DECLARE_DAAL_STRING_CONST(logTheta)                          \
    DECLARE_DAAL_STRING_CONST(tableToFill)                       \
    DECLARE_DAAL_STRING_CONST(randomNumbers)                     \
    DECLARE_DAAL_STRING_CONST(logP)                              \
    DECLARE_DAAL_STRING_CONST(coefficients)                      \
    DECLARE_DAAL_STRING_CONST(p)                                 \
    DECLARE_DAAL_STRING_CONST(auxCoefficients)                   \
    DECLARE_DAAL_STRING_CONST(auxNumberOfCoefficients)           \
    DECLARE_DAAL_STRING_CONST(shrinkingStep)                     \
    DECLARE_DAAL_STRING_CONST(shrinkage)                         \
    DECLARE_DAAL_STRING_CONST(transformedData)                   \
    DECLARE_DAAL_STRING_CONST(classSize)                         \
    DECLARE_DAAL_STRING_CONST(groupSum)                          \
    DECLARE_DAAL_STRING_CONST(auxIntermediateValue)              \
    DECLARE_DAAL_STRING_CONST(numberOfModels)                    \
    DECLARE_DAAL_STRING_CONST(explainedVariances)                \
    DECLARE_DAAL_STRING_CONST(explainedVariancesRatios)          \
    DECLARE_DAAL_STRING_CONST(inputCollection)                   \
    DECLARE_DAAL_STRING_CONST(nFeatures)                         \
    DECLARE_DAAL_STRING_CONST(noiseVariance)                     \
    DECLARE_DAAL_STRING_CONST(minObservations)                   \
    DECLARE_DAAL_STRING_CONST(nBins)                             \
    DECLARE_DAAL_STRING_CONST(resultsToEvaluate)                 \
    DECLARE_DAAL_STRING_CONST(coreIndices)                       \
    DECLARE_DAAL_STRING_CONST(coreObservations)                  \
    DECLARE_DAAL_STRING_CONST(blockIndex)                        \
    DECLARE_DAAL_STRING_CONST(nBlocks)                           \
    DECLARE_DAAL_STRING_CONST(leftBlocks)                        \
    DECLARE_DAAL_STRING_CONST(rightBlocks)                       \
    DECLARE_DAAL_STRING_CONST(partialWeights)                    \
    DECLARE_DAAL_STRING_CONST(step1Data)                         \
    DECLARE_DAAL_STRING_CONST(partialOrder)                      \
    DECLARE_DAAL_STRING_CONST(boundingBox)                       \
    DECLARE_DAAL_STRING_CONST(step3PartialBoundingBoxes)         \
    DECLARE_DAAL_STRING_CONST(split)                             \
    DECLARE_DAAL_STRING_CONST(step4PartialSplits)                \
    DECLARE_DAAL_STRING_CONST(step4PartialOrders)                \
    DECLARE_DAAL_STRING_CONST(partitionedData)                   \
    DECLARE_DAAL_STRING_CONST(partitionedWeights)                \
    DECLARE_DAAL_STRING_CONST(partitionedPartialOrders)          \
    DECLARE_DAAL_STRING_CONST(step5PartialBoundingBoxes)         \
    DECLARE_DAAL_STRING_CONST(partitionedHaloData)               \
    DECLARE_DAAL_STRING_CONST(partitionedHaloDataIndices)        \
    DECLARE_DAAL_STRING_CONST(partitionedHaloWeights)            \
    DECLARE_DAAL_STRING_CONST(haloData)                          \
    DECLARE_DAAL_STRING_CONST(haloDataIndices)                   \
    DECLARE_DAAL_STRING_CONST(haloWeights)                       \
    DECLARE_DAAL_STRING_CONST(haloBlocks)                        \
    DECLARE_DAAL_STRING_CONST(step6ClusterStructure)             \
    DECLARE_DAAL_STRING_CONST(step6FinishedFlag)                 \
    DECLARE_DAAL_STRING_CONST(step6NClusters)                    \
    DECLARE_DAAL_STRING_CONST(step6Queries)                      \
    DECLARE_DAAL_STRING_CONST(partialFinishedFlags)              \
    DECLARE_DAAL_STRING_CONST(finishedFlag)                      \
    DECLARE_DAAL_STRING_CONST(step8InputClusterStructure)        \
    DECLARE_DAAL_STRING_CONST(step8InputNClusters)               \
    DECLARE_DAAL_STRING_CONST(step8PartialQueries)               \
    DECLARE_DAAL_STRING_CONST(step8ClusterStructure)             \
    DECLARE_DAAL_STRING_CONST(step8FinishedFlag)                 \
    DECLARE_DAAL_STRING_CONST(step8NClusters)                    \
    DECLARE_DAAL_STRING_CONST(step8Queries)                      \
    DECLARE_DAAL_STRING_CONST(partialNClusters)                  \
    DECLARE_DAAL_STRING_CONST(step9NClusters)                    \
    DECLARE_DAAL_STRING_CONST(clusterOffsets)                    \
    DECLARE_DAAL_STRING_CONST(step10InputClusterStructure)       \
    DECLARE_DAAL_STRING_CONST(step10ClusterOffset)               \
    DECLARE_DAAL_STRING_CONST(step10ClusterStructure)            \
    DECLARE_DAAL_STRING_CONST(step10FinishedFlag)                \
    DECLARE_DAAL_STRING_CONST(step10Queries)                     \
    DECLARE_DAAL_STRING_CONST(step11InputClusterStructure)       \
    DECLARE_DAAL_STRING_CONST(step11PartialQueries)              \
    DECLARE_DAAL_STRING_CONST(step11ClusterStructure)            \
    DECLARE_DAAL_STRING_CONST(step11FinishedFlag)                \
    DECLARE_DAAL_STRING_CONST(step11Queries)                     \
    DECLARE_DAAL_STRING_CONST(step12InputClusterStructure)       \
    DECLARE_DAAL_STRING_CONST(step12PartialOrders)               \
    DECLARE_DAAL_STRING_CONST(assignmentQueries)                 \
    DECLARE_DAAL_STRING_CONST(partialAssignmentQueries)          \
    DECLARE_DAAL_STRING_CONST(step13Assignments)                 \
    DECLARE_DAAL_STRING_CONST(step13AssignmentQueries)           \
    DECLARE_DAAL_STRING_CONST(gramMatrix)                        \
    DECLARE_DAAL_STRING_CONST(lassoParameters)

/**
 *  oneAPI Data Analytics Library namespace
 */
namespace daal
{
#define DECLARE_DAAL_STRING_CONST(arg1) const char * arg1##Str();
DAAL_STRINGS_LIST()
#undef DECLARE_DAAL_STRING_CONST

#define DECLARE_DAAL_STRING_CONST(arg1) arg1##EStr,
enum EStringConst
{
    DAAL_STRINGS_LIST() eLastStringConst
};
#undef DECLARE_DAAL_STRING_CONST

const char * getStr(EStringConst eStr);
}; // namespace daal
/** @} */

#endif
