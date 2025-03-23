/* file: error_indexes.h */
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
//  Details of errors in oneAPI Data Analytics Library (oneDAL).
//--
*/

#ifndef __ERROR_INDEXES__
#define __ERROR_INDEXES__

namespace daal
{
namespace services
{
/**
 * @defgroup error_handling Handling Errors
 * \brief Contains classes and methods to handle exceptions or errors that can occur during library operation.
 * @ingroup services
 * @{
 */
/**
 * <a name="DAAL-ENUM-SERVICES__ERRORDETAILID"></a>
 * Available error detail ID to represent additional information in error description
 */
enum ErrorDetailID
{
    NoErrorMessageDetailFound = 0,  /*!< No error message detail found */
    Row                       = 1,  /*!< Detail about row */
    Column                    = 2,  /*!< Detail about column */
    Rank                      = 3,  /*!< Detail about rank */
    StatisticsName            = 4,  /*!< Detail about statistics name function */
    Method                    = 5,  /*!< Detail about method */
    Iteration                 = 6,  /*!< Detail about iteration number */
    Component                 = 7,  /*!< Detail about component number */
    Minor                     = 8,  /*!< Detail about order of matrix minor */
    ArgumentName              = 9,  /*!< Detail about argument name */
    ElementInCollection       = 10, /*!< Detail about element in collection */
    Dimension                 = 11, /*!< Detail about tensor dimension */
    ParameterName             = 12, /*!< Detail about parameter name */
    OptionalInput             = 13, /*!< Detail about optional input name */
    OptionalResult            = 14, /*!< Detail about optional result name */
    SerializationTag          = 16, /*!< Detail about serialization tag */
    ExpectedValue             = 17, /*!< Detail about expected value */
    ActualValue               = 18, /*!< Detail about actual value */
    Sycl                      = 19, /*!< Detail about Sycl */
    OpenCL                    = 20, /*!< Detail about actual OpenCL */
    LevelZero                 = 21, /*!< Detail about actual LevelZero */
    Key                       = 22  /*!< Detail about key */
};

/**
 * <a name="DAAL-ENUM-SERVICES__ERRORID"></a>
 * Execution statuses
 */
enum ErrorID
{
    // Input errors: -1..-1999
    ErrorMethodNotSupported                           = -1,  /*!< Method not supported by the algorithm */
    ErrorIncorrectNumberOfFeatures                    = -3,  /*!< Number of columns in numeric table is incorrect */
    ErrorIncorrectNumberOfObservations                = -4,  /*!< Number of rows in numeric table is incorrect */
    ErrorIncorrectSizeOfArray                         = -7,  /*!< Incorrect size of array */
    ErrorNullParameterNotSupported                    = -8,  /*!< Null parameter is not supported by the algorithm */
    ErrorIncorrectNumberOfArguments                   = -9,  /*!< Number of arguments is incorrect */
    ErrorIncorrectInputNumericTable                   = -10, /*!< Input numeric table is incorrect */
    ErrorEmptyInputNumericTable                       = -11, /*!< Input numeric table is empty */
    ErrorIncorrectDataRange                           = -12, /*!< Data range is incorrect */
    ErrorPrecomputedStatisticsIndexOutOfRange         = -13, /*!< Precomputed statistics index is out of range */
    ErrorIncorrectNumberOfInputNumericTables          = -14, /*!< Incorrect number of input numeric tables */
    ErrorIncorrectNumberOfOutputNumericTables         = -15, /*!< Incorrect number of output numeric tables */
    ErrorNullInputNumericTable                        = -16, /*!< Null input numeric table is not supported */
    ErrorNullOutputNumericTable                       = -17, /*!< Null output numeric table is not supported */
    ErrorNullModel                                    = -18, /*!< Null model is not supported */
    ErrorInconsistentNumberOfRows                     = -19, /*!< Number of rows in provided numeric tables is inconsistent */
    ErrorIncorrectSizeOfInputNumericTable             = -20, /*!< Number of columns or rows in input numeric table is incorrect */
    ErrorIncorrectSizeOfOutputNumericTable            = -21, /*!< Number of columns or rows in output numeric table is incorrect */
    ErrorIncorrectNumberOfRowsInInputNumericTable     = -24, /*!< Number of rows in input numeric table is incorrect */
    ErrorIncorrectNumberOfColumnsInInputNumericTable  = -25, /*!< Number of columns in input numeric table is incorrect */
    ErrorIncorrectNumberOfRowsInOutputNumericTable    = -26, /*!< Number of rows in output numeric table is incorrect */
    ErrorIncorrectNumberOfColumnsInOutputNumericTable = -27, /*!< Number of columns in output numeric table is incorrect */
    ErrorIncorrectTypeOfInputNumericTable             = -28, /*!< Incorrect type of input NumericTable */
    ErrorIncorrectTypeOfOutputNumericTable            = -29, /*!< Incorrect type of output NumericTable */
    ErrorIncorrectNumberOfElementsInInputCollection   = -30, /*!< Incorrect number of elements in input collection */
    ErrorIncorrectNumberOfElementsInResultCollection  = -31, /*!< Incorrect number of elements in result collection */
    ErrorNullInput                                    = -32, /*!< Input not set */
    ErrorNullResult                                   = -33, /*!< Result not set */
    ErrorIncorrectParameter                           = -34, /*!< Incorrect parameter */
    ErrorModelNotFullInitialized                      = -35, /*!< Model is not full initialized */
    ErrorInconsistentNumberOfColumns                  = -36, /*!< Inconsistent number of rows in Numeric Table*/
    ErrorIncorrectIndex                               = -37, /*!< Index in collection is out of range */
    ErrorDataArchiveInternal                          = -38, /*!< Incorrect size of data block */
    ErrorNullPartialModel                             = -39, /*!< Null partial model is not supported */
    ErrorNullInputDataCollection                      = -40, /*!< Null input data collection is not supported */
    ErrorNullOutputDataCollection                     = -41, /*!< Null output data collection is not supported */
    ErrorNullPartialResult                            = -42, /*!< Partial result not set */
    ErrorNullLayerData                                = -48, /*!< Null layer data is not supported */
    ErrorIncorrectSizeOfLayerData                     = -49, /*!< Incorrect number of elements in layer data collection */
    ErrorNullNumericTable                             = -50, /*!< Null numeric table is not supported */
    ErrorIncorrectNumberOfColumns                     = -51, /*!< Number of columns in numeric table is incorrect */
    ErrorIncorrectNumberOfRows                        = -52, /*!< Number of rows in numeric table is incorrect */
    ErrorIncorrectTypeOfNumericTable                  = -53, /*!< Incorrect type of Numeric Table */
    ErrorUnsupportedCSRIndexing                       = -54, /*!< CSR Numeric Table has unsupported indexing type */
    ErrorSignificanceLevel                            = -55, /*!< Incorrect significance level value */
    ErrorAccuracyThreshold                            = -56, /*!< Incorrect accuracy threshold */
    ErrorIncorrectNumberOfBetas                       = -57, /*!< Incorrect number of betas in linear regression model */
    ErrorIncorrectNumberOfBetasInReducedModel         = -58, /*!< Incorrect number of betas in reduced linear regression model */
    ErrorNumericTableIsNotSquare                      = -59, /*!< Numeric table is not square */
    ErrorNullAuxiliaryAlgorithm                       = -60, /*!< Null auxiliary algorithm */
    ErrorNullInitializationProcedure                  = -61, /*!< Null initialization procedure */
    ErrorNullAuxiliaryDataCollection                  = -62, /*!< Null auxiliary data collection */
    ErrorEmptyAuxiliaryDataCollection                 = -63, /*!< Empty auxiliary data collection */
    ErrorIncorrectElementInCollection                 = -64, /*!< Incorrect element in collection */
    ErrorNullPartialResultDataCollection              = -65, /*!< Null partial result data collection */
    ErrorIncorrectElementInPartialResultCollection    = -66, /*!< Incorrect element in collection of partial results */
    ErrorIncorrectElementInNumericTableCollection     = -67, /*!< Incorrect element in collection of numeric tables */
    ErrorNullOptionalResult                           = -68, /*!< Null optional result */
    ErrorIncorrectOptionalResult                      = -69, /*!< Incorrect optional result */
    ErrorIncorrectOptionalInput                       = -70, /*!< Incorrect optional input */
    ErrorIncorrectNumberOfPartialClusters             = -71, /*!< Incorrect number of partial clusters */
    ErrorIncorrectTotalNumberOfPartialClusters        = -72, /*!< Incorrect total number of partial clusters */
    ErrorIncorrectDataCollectionSize                  = -73, /*!< Incorrect DataCollection size*/
    ErrorIncorrectValueInTheNumericTable              = -74, /*!< Incorrect value in the numeric table */
    ErrorIncorrectItemInDataCollection                = -75, /*!< Incorrect item in data collection */
    ErrorNullPtr                                      = -76, /*!< Null pointer in input arguments */
    ErrorUndefinedFeature                             = -77, /*!< Dictionary contains a undefined feature */
    ErrorCloneMethodFailed                            = -78, /*!< Cloning of algorithm failed */
    ErrorDataTypeNotSupported                         = -79, /*!< Data type not supported */
    ErrorBufferSizeIntegerOverflow                    = -80, /*!< Integer oveflow is occured during buffer size calculation */
    ErrorHyperparameterNotFound                       = -81, /*!< Cannot find a hyperparameter with a given id */
    ErrorHyperparameterCanNotBeSet                    = -82, /*!< Cannot set a hyperparameter with a specified id */
    ErrorHyperparameterBadValue                       = -83, /*!< Provided a bad value for a hyperparameter */

    // Environment errors: -2000..-2999
    ErrorCpuNotSupported          = -2000, /*!< CPU not supported */
    ErrorMemoryAllocationFailed   = -2001, /*!< Memory allocation failed */
    ErrorEmptyDataBlock           = -2004, /*!< Empty data block */
    ErrorMemoryCopyFailedInternal = -2005, /*!< Memory copy internal error */
    ErrorCpuIsInvalid             = -2006, /*!< Invalid CPU value used */

    // Workflow errors: -3000..-3999
    ErrorIncorrectCombinationOfComputationModeAndStep = -3002, /*!< Incorrect combination of computation mode and computation step */
    ErrorDictionaryAlreadyAvailable                   = -3003, /*!< Data Dictionary is already available */
    ErrorDictionaryNotAvailable                       = -3004, /*!< Data Dictionary is not available */
    ErrorNumericTableNotAvailable                     = -3005, /*!< Numeric Table is not available */
    ErrorNumericTableAlreadyAllocated                 = -3006, /*!< Numeric Table was already allocated */
    ErrorNumericTableNotAllocated                     = -3007, /*!< Numeric Table is not allocated */
    ErrorPrecomputedSumNotAvailable                   = -3008, /*!< Precomputed sums are not available */
    ErrorPrecomputedMinNotAvailable                   = -3009, /*!< Precomputed minimum values are not available */
    ErrorPrecomputedMaxNotAvailable                   = -3010, /*!< Precomputed maximum values are not available */
    ErrorServiceMicroTableInternal                    = -3011, /*!< Numeric Table internal error */
    ErrorEmptyCSRNumericTable                         = -3012, /*!< CSR Numeric Table is empty */
    ErrorEmptyHomogenNumericTable                     = -3013, /*!< Homogeneous Numeric Table is empty */
    ErrorSourceDataNotAvailable                       = -3014, /*!< Source data is not available */
    ErrorEmptyDataSource                              = -3015, /*!< Data source is empty */
    ErrorIncorrectClassLabels                         = -3016, /*!< Class labels provided to classification algorithm are incorrect */
    ErrorIncorrectSizeOfModel                         = -3017, /*!< Incorrect size of model */
    ErrorIncorrectTypeOfModel                         = -3018, /*!< Incorrect type of model */
    ErrorIncorrectErrorcodeFromGenerator              = -3019, /*!< Incorrect error code is returned from data generator */
    ErrorLeapfrogUnsupported                          = -3020, /*!< Leapfrog method is not supported by generator */
    ErrorSkipAheadUnsupported                         = -3021, /*!< SkipAhead method is not supported by generator */
    ErrorFeatureNamesNotAvailable                     = -3022, /*!< Feature names are not available for feature modifier */
    ErrorEngineNotSupported                           = -3023,

    // Common computation errors: -4000...
    ErrorInputSigmaMatrixHasNonPositiveMinor = -4001, /*!< Input sigma matrix has non positive minor */
    ErrorInputSigmaMatrixHasIllegalValue     = -4002, /*!< Input sigma matrix has illegal value */
    ErrorIncorrectInternalFunctionParameter  = -4003, /*!< Incorrect parameter in internal function call */
    ErrorUserCancelled                       = -4004, /*!< Computation cancelled at user's request */

    /* Apriori algorithm errors -5000..-5199 */
    ErrorAprioriIncorrectItemsetTableSize    = -5000, /*!< Number of rows in the output table containing
                                                                         *   'large' item sets is too small */
    ErrorAprioriIncorrectSupportTableSize    = -5001, /*!< Number of rows in the output table containing
                                                                         *   'large' item sets support values is too small */
    ErrorAprioriIncorrectLeftRuleTableSize   = -5002, /*!< Number of rows in the output table containing
                                                                         *   left parts of the association rules is too small */
    ErrorAprioriIncorrectRightRuleTableSize  = -5003, /*!< Number of rows in the output table containing
                                                                         *   right parts of the association rules is too small */
    ErrorAprioriIncorrectConfidenceTableSize = -5004, /*!< Number of rows in the output table containing
                                                                         *   association rules confidence is too small */
    ErrorAprioriIncorrectInputData           = -5005, /*!< Incorrect input data */

    // Boosting errors: -5200..-5399
    ErrorInconsistentNumberOfClasses = -5200, /*!< Inconsistent number of classes between boosting
                                                                         *   algorithm and weak learner */

    // Cholesky errors: -5400..-5599
    ErrorCholeskyInternal               = -5400, /*!< Cholesky internal error */
    ErrorInputMatrixHasNonPositiveMinor = -5401, /*!< Input matrix has non positive minor */

    // Covariance errors: -5600..-5799
    ErrorCovarianceInternal = -5600, /*!< Covariance internal error */

    // Distance errors: -5800..-5999

    // EM errors: -6000..-6099
    ErrorEMMatrixInverse                   = -6001, /*!< Sigma matrix on M-step cannot be inverted */
    ErrorEMIncorrectToleranceToConverge    = -6002, /*!< Incorrect value of tolerance to converge in EM parameter */
    ErrorEMIllConditionedCovarianceMatrix  = -6003, /*!< Ill-conditioned covariance matrix */
    ErrorEMIncorrectMaxNumberOfIterations  = -6004, /*!< Incorrect maximum number of iterations value in EM parameter */
    ErrorEMNegativeDefinedCovarianceMartix = -6005, /*!< Negative-defined covariance matrix */
    ErrorEMEmptyComponent                  = -6006, /*!< Empty component during computation */
    ErrorEMCovariance                      = -6007, /*!< Error during covariance computation for component on M step */
    ErrorEMIncorrectNumberOfComponents     = -6008, /*!< Incorrect number of components value in EM parameter */

    // EM initialization errors: -6100..-6199
    ErrorEMInitNoTrialConverges               = -6100, /*!< No trial of internal EM start converges */
    ErrorEMInitIncorrectToleranceToConverge   = -6101, /*!< Incorrect tolerance to converge value in EM initialization parameter */
    ErrorEMInitIncorrectDepthNumberIterations = -6102, /*!< Incorrect depth number of iterations value in EM init parameter */
    ErrorEMInitIncorrectNumberOfTrials        = -6103, /*!< Incorrect number of trials value in EM initialization parameter */
    ErrorEMInitIncorrectNumberOfComponents    = -6104, /*!< Incorrect number of components value in EM initialization parameter */
    ErrorEMInitInconsistentNumberOfComponents = -6105, /*!< Inconsistent number of component: number of observations should be
                                                                             greater than number of components */
    ErrorVarianceComputation                  = -6106, /*!< Error during variance computation */

    // KernelFunction errors: -6200..-6399

    // KMeans errors: -6400..-6599
    ErrorKMeansNumberOfClustersIsTooLarge = -6400, /*!< Number of clusters exceeds the number of points */

    // Linear Rergession errors: -6600..-6799
    ErrorLinearRegressionInternal   = -6600, /*!< Linear Regression internal error */
    ErrorNormEqSystemSolutionFailed = -6601, /*!< Failed to solve the system of normal equations */
    ErrorLinRegXtXInvFailed         = -6602, /*!< Failed to invert Xt*X matrix */

    // LogitBoots errors: -6800..-6999

    // LowOrderMoments errors: -7000..-7199
    ErrorLowOrderMomentsInternal = -7000, /*!< Low Order Moments internal error */

    // MultiClassClassifier errors: -7200..-7399
    ErrorIncorrectNumberOfClasses                    = -7200, /*!< Number of classes provided to classifier is too small */
    ErrorMultiClassNullTwoClassTraining              = -7201, /*!< Null two-class classifier training algorithm is not supported */
    ErrorMultiClassFailedToTrainTwoClassClassifier   = -7202, /*!< Failed to train a model of two-class classifier */
    ErrorMultiClassFailedToComputeTwoClassPrediction = -7203, /*!< Failed to compute prediction based on two-class classifier model */

    // NaiveBayes errors: -7400..-7599
    ErrorEmptyInputCollection     = -7400, /*!< Naive Bayes: Input collection is empty */
    ErrorNaiveBayesIncorrectModel = -7401, /*!< Naive Bayes: Input model is not consistent with the number of classes */

    // OutlierDetection errors: -7600..-7799
    ErrorOutlierDetectionInternal = -7600, /*!< Outlier Detection internal error */

    /* PCA errors: -7800..-7999 */
    ErrorPCAFailedToComputeCorrelationEigenvalues           = -7800, /*!< Failed to compute eigenvalues of the correlation matrix */
    ErrorPCACorrelationInputDataTypeSupportsOfflineModeOnly = -7801, /*!< This type of the input data supports
                                                                         *   only offline mode of the computations */
    ErrorIncorrectCrossProductTableSize                     = -7802, /*!< Number of columns or rows
                                                                         *   in cross-product numeric table is incorrect */
    ErrorCrossProductTableIsNotSquare                       = -7803, /*!< Number of columns or rows
                                                                         *   in cross-product numeric table is not equal */
    ErrorInputCorrelationNotSupportedInOnlineAndDistributed = -7804, /*!< Input correlation matrix is not supported in online and distributed
                                                                             computation modes */
    ErrorIncorrectNComponents                               = -7805, /*!< Incorrect nComponents parameter: nComponents should be less or equal
                                                                             to number of columns in testing dataset */
    ErrorIncorrectEigenValuesSum                            = -7806, /*!< The sum of eigenvalues is less or equal to zero */
    ErrorIncorrectSingularValuesDenominator                 = -7807, /*!< The denominator of eigenvalues is less or equal to zero */

    // QR errors: -8000..-8199
    ErrorQRInternal             = -8000, /*!< QR internal error */
    ErrorQrIthParamIllegalValue = -8001, /*!< QR internal error */
    ErrorQrXBDSQRDidNotConverge = -8002, /*!< QR internal error */

    // Stump errors: -8200..-8299
    ErrorStumpIncorrectSplitFeature       = -8200, /*!< Incorrect split feature  */
    ErrorStumpInvalidInputCategoricalData = -8201, /*!< Invalid stump training data: all features in the input table are categorical
                                                                             and each feature has < 2 categories */

    // SVD errors: -8300..-8399
    ErrorSvdIthParamIllegalValue = -8300, /*!< SVD internal error */
    ErrorSvdXBDSQRDidNotConverge = -8301, /*!< SVD internal error */

    // LCN errors: -8400..-8599
    ErrorLCNinnerConvolution = -8400, /*!< Error in convolution 2d layer  */

    // SVM errors: -8600..-8799
    ErrorSVMPredictKernerFunctionCall = -8601, /*!< SVM predict: error in kernel function call. Details are as follows. */

    // WeakLearner errors: -8800..-8999
    ErrorIncorrectWeakLearnerClassificationAlgorithm = -8800, /*!< Weak learner can not be casted to classifier algorithm */
    ErrorIncorrectWeakLearnerRegressionAlgorithm     = -8801, /*!< Weak learner can not be casted to regression algorithm */
    ErrorIncorrectWeakLearnerClassificationModel     = -8802, /*!< Weak learner's model can not be casted to classifier model */
    ErrorIncorrectWeakLearnerRegressionModel         = -8803, /*!< Weak learner's model can not be casted to regression model */

    // Compression errors: -9000..-9199
    ErrorCompressionNullInputStream  = -9000, /*!< Null input stream is not supported */
    ErrorCompressionNullOutputStream = -9001, /*!< Null output stream is not supported */

    ErrorCompressionEmptyInputStream  = -9002, /*!< Input stream of size 0 is not supported */
    ErrorCompressionEmptyOutputStream = -9003, /*!< Output stream of size 0 is not supported */

    ErrorZlibInternal               = -9004, /*!< Zlib internal error */
    ErrorZlibDataFormat             = -9005, /*!< Input compressed stream is in wrong format,
                                                                         *   corrupted or contains not a whole number of compressed blocks */
    ErrorZlibParameters             = -9006, /*!< Unsupported Zlib parameters */
    ErrorZlibMemoryAllocationFailed = -9007, /*!< Internal Zlib memory allocation failed */
    ErrorZlibNeedDictionary         = -9008, /*!< Specific dictionary is needed for decompression,
                                                                         *   currently unsupported Zlib feature */

    ErrorBzip2Internal               = -9009, /*!< Bzip2 internal error */
    ErrorBzip2DataFormat             = -9010, /*!< Input compressed stream is in wrong format,
                                                                         *   corrupted or contains not a whole number of compressed blocks */
    ErrorBzip2Parameters             = -9011, /*!< Unsupported Bzip2 parameters */
    ErrorBzip2MemoryAllocationFailed = -9012, /*!< Internal Bzip2 memory allocation failed */

    ErrorLzoInternal                    = -9013, /*!< LZO internal error */
    ErrorLzoOutputStreamSizeIsNotEnough = -9014, /*!< Size of output stream is not enough to start compression */
    ErrorLzoDataFormat                  = -9015, /*!< Input compressed stream is in wrong format or corrupted */
    ErrorLzoDataFormatLessThenHeader    = -9016, /*!< Size of input compressed stream is less then
                                                                         *   compressed block header size */
    ErrorLzoDataFormatNotFullBlock      = -9017, /*!< Input compressed stream contains not a whole
                                                                         *   number of compressed blocks */

    ErrorRleInternal                    = -9018, /*!< RLE internal error */
    ErrorRleOutputStreamSizeIsNotEnough = -9019, /*!< Size of output stream is not enough to start compression */
    ErrorRleDataFormat                  = -9020, /*!< Input compressed stream is in wrong format or corrupted */
    ErrorRleDataFormatLessThenHeader    = -9021, /*!< Size of input compressed stream is less then
                                                                         *   compressed block header size */
    ErrorRleDataFormatNotFullBlock      = -9022, /*!< Input compressed stream contains not a whole
                                                                         *   number of compressed blocks */
    // Min-max normalization errors: -9400..-9499
    ErrorLowerBoundGreaterThanOrEqualToUpperBound = -9400, /*!< Lower bound parameter greater than or equal to upper bound */

    // Quantile error: -10000..-11000
    ErrorQuantileOrderValueIsInvalid = -10001, /*!< Quantile order value is invalid */
    ErrorQuantilesInternal           = -10002, /*!< Quantile internal error */

    // ALS errors: -11000..-12000
    ErrorALSInternal                     = -11000, /*!< ALS algorithm failed to solve a system of normal equations */
    ErrorALSInconsistentSparseDataBlocks = -11001, /*!< Failed to find a non-zero value with needed indices
                                                                             in a sparse data block */
    // Sorting error: -12000..-13000
    ErrorSorting = -12001, /*!< Cannot sort the numeric table */

    // SGD error: -13000..-14000
    ErrorNegativeLearningRate = -13000, /*!< Negative learning rate */

    // Normalization errors: -14000..-15000
    ErrorMeanAndStandardDeviationComputing = -14000, /*!< Computation of mean and standard deviation failed */
    ErrorNullVariance                      = -14001, /*!< Failed to normalize data in column: it has null variance deviation */
    ErrorMinAndMaxComputing                = -14002, /*!< Computation of minimum and maximum failed */

    //Sum of functions error: -15000..-16000
    ErrorZeroNumberOfTerms = -15000, /*!< Number of terms can not be zero */

    //Covolution layer error: -16000..-17000
    ErrorConvolutionInternal  = -16000, /*!< Convoltion internal error */
    ErrorIncorrectKernelSise1 = -16001, /*!< Convolution2d backward: incorrect parameter kernelSize1 */
    ErrorIncorrectKernelSise2 = -16002, /*!< Convolution2d backward: incorrect parameter kernelSize2 */

    // Ridge Regression errors: -17000..-17999
    ErrorRidgeRegressionInternal                   = -17000, /*!< Ridge Regression internal error */
    ErrorRidgeRegressionNormEqSystemSolutionFailed = -17001, /*!< Failed to solve the system of normal equations */
    ErrorRidgeRegressionInvertFailed               = -17002, /*!< Failed to invert matrix */

    // Pivoted QR errors: -19000..-19199
    ErrorPivotedQRInternal = -19000, /*!< Pivoted QR internal error */

    // Decision forest error: -20000..-20099
    ErrorDFBootstrapVarImportanceIncompatible = -20000, /*!< Parameter 'bootstrap' is incompatible with requested variable importance type */
    ErrorDFBootstrapOOBIncompatible = -20001, /*!< Parameter 'bootstrap' is incompatible with requested OOB result (no out-of-bag observations) */

    // K-Nearest Neighbors errors: -21000..21999
    ErrorKNNInternal = -21000, /*!< K-Nearest Neighbors internal error */

    // GBT error: -30000..-30099
    ErrorGbtIncorrectNumberOfTrees             = -30000, /*!< Number of trees in the model is not consistent with the number of classes */
    ErrorGbtPredictIncorrectNumberOfIterations = -30001, /*!< Number of iterations value in GBT parameter is not consistent with the model */
    ErrorGbtPredictShapOptions                 = -30002, /*!< For SHAP values, calculate either contributions or interactions, not both */
    ErrorGbtPredictShapMulticlassNotSupported  = -30003, /*!< For classification, SHAP values currently only support binary classification */

    // Data management errors:  -80001..
    ErrorUserAllocatedMemory = -80001, /*!< Couldn't free memory allocated by user */

    //Math errors: -90000..-100000
    ErrorDataSourseNotAvailable = -90041, /*!< ErrorDataSourseNotAvailable */
    ErrorHandlesSQL             = -90042, /*!< ErrorHandlesSQL */
    ErrorODBC                   = -90043, /*!< ErrorODBC */
    ErrorSQLstmtHandle          = -90044, /*!< ErrorSQLstmtHandle */
    ErrorOnFileOpen             = -90045, /*!< Error on file open */
    ErrorOnFileRead             = -90046, /*!< Error on file read */
    ErrorNullByteInjection      = -90047, /*!< Error null byte injection */

    ErrorKDBNoConnection      = -90051, /*!< ErrorKDBNoConnection */
    ErrorKDBWrongCredentials  = -90052, /*!< ErrorKDBWrongCredentials */
    ErrorKDBNetworkError      = -90053, /*!< ErrorKDBNetworkError */
    ErrorKDBServerError       = -90054, /*!< ErrorKDBServerError */
    ErrorKDBTypeUnsupported   = -90055, /*!< ErrorKDBTypeUnsupported */
    ErrorKDBWrongTypeOfOutput = -90056, /*!< ErrorKDBWrongTypeOfOutput */

    ErrorIncorrectEngineParameter = -90100, /*!< Incorrect engine parameter in distribution */

    // Quality metrics errors -90201..-90301
    ErrorEmptyInputAlgorithmsCollection = -90201, /*!< Input algorithms collection is empty */

    // Group of SYCL-related errors -90900..-90999
    ErrorEmptyBuffer                   = -90900, /*!< Buffer is empty */
    ErrorAccessUSMPointerOnOtherDevice = -90901, /*!< Cannot access USM pointer from the other device */

    // Other errors: -100000..
    ErrorObjectDoesNotSupportSerialization = -100000, /*!< SerializationIface is not implemented or implemented incorrectly */
    ErrorExecutionContext                  = -100001, /*!< Execution context errors*/
    ErrorHashTableCollision                = -100002, /*!< Hash table collision happened. Please, increase the buffer size.*/

    ErrorCouldntAttachCurrentThreadToJavaVM       = -110001, /*!< Couldn't attach current thread to Java VM */
    ErrorCouldntCreateGlobalReferenceToJavaObject = -110002, /*!< Couldn't create global reference to Java object */
    ErrorCouldntFindJavaMethod                    = -110003, /*!< Couldn't find Java method */
    ErrorCouldntFindClassForJavaObject            = -110004, /*!< Couldn't find class for Java object */
    ErrorCouldntDetachCurrentThreadFromJavaVM     = -110005, /*!< Couldn't detach current thread from Java VM */

    UnknownError              = -1000000, /*!< Unknown error */
    NoErrorMessageFound       = -1000001, /*!< No error message found */
    ErrorMethodNotImplemented = -1000002, /*!< Method is not implemented in the present library version  */

    ErrorIncorrectOffset                               = -1000003, /*!< Incorrect offset */
    ErrorIterativeSolverIncorrectMaxNumberOfIterations = -1000004, /*!< Incorrect maximum number of iterations value in solver */
    ErrorIncorrectNumberOfTerms                        = -1000005, /*!< Incorrect number of summands (terms) in objective function */
    ErrorIncorrectNumberOfNodes                        = -1000006, /*!< Incorrect number of nodes */

    ErrorDeviceSupportNotImplemented               = -1000007, /*!< GPU support for the method is not implemented in the present library version  */
    ErrorInconsistenceModelAndBatchSizeInParameter = -1000008, /*!< Inconsistence of model and batch size parameter in optimization solver */

    ErrorCanNotLoadDynamicLibrary       = -1000009, /*!< Failure during loading of dynamic library */
    ErrorCanNotLoadDynamicLibrarySymbol = -1000010  /*!< Failure during loading symbol from dynamic library */
};
/** @} */
} // namespace services
} // namespace daal

#endif
