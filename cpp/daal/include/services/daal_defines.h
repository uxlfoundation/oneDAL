/* file: daal_defines.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/*
//++
//  Common definitions.
//--
*/

#ifndef __DAAL_DEFINES_H__
#define __DAAL_DEFINES_H__

/** \file daal_defines.h */

#include <cstddef> // for size_t

#if defined(__x86_64__) || defined(__x86_64) || defined(__amd64) || defined(_M_AMD64)
    #define TARGET_X86_64
#endif

#if defined(__ARM_ARCH) || defined(__aarch64__)
    #define TARGET_ARM
#endif

#if defined(__riscv) && (__riscv_xlen == 64)
    #define TARGET_RISCV64
#endif

#if (defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)) && !defined(SYCL_LANGUAGE_VERSION)
    #define DAAL_INTEL_CPP_COMPILER
#endif

#if !(defined(_MSC_VER) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER))
    #define __int32 int
    #define __int64 long long int
#endif

#ifdef __DAAL_IMPLEMENTATION
    #if defined(_WIN32) || defined(_WIN64)
        #define DAAL_EXPORT __declspec(dllexport)
    #else
        #define DAAL_EXPORT __attribute__((visibility("default")))
    #endif
#else
    #define DAAL_EXPORT
#endif

#if (defined(__INTEL_CXX11_MODE__) || __cplusplus > 199711L)
    #define DAAL_C11_OVERRIDE override
#else
    #define DAAL_C11_OVERRIDE
#endif

/* oneDAL 64-bit integer types */
#if !(defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)) && defined(_MSC_VER)
    #define DAAL_INT64  __int64
    #define DAAL_UINT64 unsigned __int64
#else
    #define DAAL_INT64  long long int
    #define DAAL_UINT64 unsigned long long int
#endif

#if !defined(DAAL_INT)
    #if defined(_WIN64) || defined(__x86_64__)
        #define DAAL_INT __int64
    #elif defined(TARGET_ARM)
        #define DAAL_INT __int64
    #elif defined(TARGET_RISCV64)
        #define DAAL_INT __int64
    #else
        #define DAAL_INT __int32
    #endif
#endif

#if defined(DAAL_HIDE_DEPRECATED)
    #define DAAL_DEPRECATED
#else
    #ifdef __GNUC__
        #define DAAL_DEPRECATED __attribute__((deprecated))
    #elif defined(_MSC_VER)
        #define DAAL_DEPRECATED __declspec(deprecated)
    #else
        #define DAAL_DEPRECATED
    #endif
#endif

#if defined(DAAL_HIDE_DEPRECATED)
    #define DAAL_DEPRECATED_VIRTUAL
#else
    #ifdef __INTEL_COMPILER
        #define DAAL_DEPRECATED_VIRTUAL DAAL_DEPRECATED
    #else
        #define DAAL_DEPRECATED_VIRTUAL
    #endif
#endif

#if defined(_MSC_VER)
    #define DAAL_FORCEINLINE __forceinline
#else
    #define DAAL_FORCEINLINE inline __attribute__((always_inline))
#endif

#if defined(__APPLE__)
    #define DAAL_CPU_TOPO_DISABLED
    #define DAAL_THREAD_PINNING_DISABLED
#endif

#if defined(DAAL_CPU_TOPO_DISABLED)
    #define DAAL_THREAD_PINNING_DISABLED
#endif

#if !(defined(__linux__) || defined(_WIN64))
    #define DAAL_DISABLE_LEVEL_ZERO
#endif

/**
 *  oneAPI Data Analytics Library namespace
 */
namespace daal
{
/**
* <a name="DAAL-ENUM-COMPUTEMODE"></a>
* Computation modes of oneAPI Data Analytics Library (oneDAL) algorithms
*/
enum ComputeMode
{
    batch       = 1, /*!< Batch processing computation mode */
    distributed = 2, /*!< Processing of data sets distributed across several devices */
    online      = 4  /*!< Online mode - processing of data sets in blocks */
};

/**
 * <a name="DAAL-ENUM-__COMPUTESTEP"></a>
 * Describes computation steps in the distributed processing mode
 */
enum ComputeStep
{
    step1Local  = 0, /*!< First step of the distributed processing mode */
    step2Master = 1, /*!< Second step of the distributed processing mode */
    step3Local  = 2, /*!< Third step of the distributed processing mode */
    step4Local  = 3, /*!< Fourth step of the distributed processing mode */

    step2Local  = 5, /*!< Second step of the distributed processing mode performed by local node*/
    step3Master = 6, /*!< Third step of the distributed processing mode performed by master node*/
    step5Master = 7, /*!< Fifth step of the distributed processing mode performed by master node*/

    step5Local  = 8,  /*!< Fifth step of the distributed processing mode performed by local node*/
    step6Local  = 9,  /*!< Sixth step of the distributed processing mode performed by local node*/
    step7Master = 10, /*!< Seventh step of the distributed processing mode performed by master node*/
    step8Local  = 11, /*!< Eighth step of the distributed processing mode performed by local node*/
    step9Master = 12, /*!< Ninth step of the distributed processing mode performed by master node*/
    step10Local = 13, /*!< Tenth step of the distributed processing mode performed by local node*/
    step11Local = 14, /*!< Eleventh step of the distributed processing mode performed by local node*/
    step12Local = 15, /*!< Twelfth step of the distributed processing mode performed by local node*/
    step13Local = 16  /*!< Thirteenth step of the distributed processing mode performed by local node*/
};

/**
 * <a name="DAAL-ENUM-MEMTYPE"></a>
 * Describes types of memory
 */
enum MemType
{
    dram   = 0, /*!< DRAM */
    mcdram = 1  /*!< Multi-Channel DRAM */
};

typedef unsigned char byte;

/**
 * <a name="DAAL-STRUCT-ISSAMETYPE"></a>
 * Indicates if the types are not equal
 */
template <class U, class V>
struct IsSameType
{
    static const bool value = false;
};

/**
 * <a name="DAAL-STRUCT-ISSAMETYPE"></a>
 * Indicates if the types are equal
 */
template <class U>
struct IsSameType<U, U>
{
    static const bool value = true;
};

const size_t DAAL_MALLOC_DEFAULT_ALIGNMENT = 64;
const size_t DAAL_DEFAULT_ALIGNMENT_MASK   = DAAL_MALLOC_DEFAULT_ALIGNMENT - 1;

const int SERIALIZATION_HOMOGEN_NT_ID             = 1000;
const int SERIALIZATION_AOS_NT_ID                 = 3000;
const int SERIALIZATION_SOA_NT_ID                 = 3001;
const int SERIALIZATION_ARROW_IMMUTABLE_NT_ID     = 3002;
const int SERIALIZATION_SYCL_SOA_NT_ID            = 3500;
const int SERIALIZATION_SYCL_CSR_NT_ID            = 3503;
const int SERIALIZATION_DATACOLLECTION_ID         = 4000;
const int SERIALIZATION_KEYVALUEDATACOLLECTION_ID = 4010;
const int SERIALIZATION_DATAFEATURE_NT_ID         = 5000;
const int SERIALIZATION_DATADICTIONARY_NT_ID      = 6000;
const int SERIALIZATION_DATADICTIONARY_DS_ID      = 6010;
const int SERIALIZATION_MATRIX_NT_ID              = 7000;
const int SERIALIZATION_SYCL_HOMOGEN_NT_ID        = 7500;
const int SERIALIZATION_CSR_NT_ID                 = 8000;
const int SERIALIZATION_PACKEDSYMMETRIC_NT_ID     = 11000;
const int SERIALIZATION_PACKEDTRIANGULAR_NT_ID    = 12000;
const int SERIALIZATION_MERGE_NT_ID               = 13000;
const int SERIALIZATION_ROWMERGE_NT_ID            = 14000;

const int SERIALIZATION_OPTIONAL_RESULT_ID = 30000;
const int SERIALIZATION_MEMORY_BLOCK_ID    = 40000;

const int SERIALIZATION_LINEAR_REGRESSION_MODELNORMEQ_ID           = 100100;
const int SERIALIZATION_LINEAR_REGRESSION_MODELQR_ID               = 100110;
const int SERIALIZATION_LINEAR_REGRESSION_PARTIAL_RESULT_ID        = 100120;
const int SERIALIZATION_LINEAR_REGRESSION_TRAINING_RESULT_ID       = 100130;
const int SERIALIZATION_LINEAR_REGRESSION_PREDICTION_RESULT_ID     = 100140;
const int SERIALIZATION_LINEAR_REGRESSION_SINGLE_BETA_RESULT_ID    = 100150;
const int SERIALIZATION_LINEAR_REGRESSION_GROUP_OF_BETAS_RESULT_ID = 100160;
const int SERIALIZATION_LASSO_REGRESSION_MODEL_ID                  = 100170;
const int SERIALIZATION_LASSO_REGRESSION_TRAINING_RESULT_ID        = 100180;
const int SERIALIZATION_LASSO_REGRESSION_PREDICTION_RESULT_ID      = 100190;
const int SERIALIZATION_ELASTIC_NET_MODEL_ID                       = 100191;
const int SERIALIZATION_ELASTIC_NET_TRAINING_RESULT_ID             = 100192;
const int SERIALIZATION_ELASTIC_NET_PREDICTION_RESULT_ID           = 100193;

const int SERIALIZATION_PCA_RESULT_ID                     = 100200;
const int SERIALIZATION_PCA_PARTIAL_RESULT_CORRELATION_ID = 100210;
const int SERIALIZATION_PCA_PARTIAL_RESULT_SVD_ID         = 100220;
const int SERIALIZATION_PCA_TRANSFORM_RESULT_ID           = 100230;
const int SERIALIZATION_PCA_QUALITY_METRIC_RESULT_ID      = 100240;

const int SERIALIZATION_STUMP_MODEL_ID                          = 100300;
const int SERIALIZATION_STUMP_TRAINING_RESULT_ID                = 100310;
const int SERIALIZATION_STUMP_CLASSIFICATION_MODEL_ID           = 100320;
const int SERIALIZATION_STUMP_CLASSIFICATION_TRAINING_RESULT_ID = 100330;
const int SERIALIZATION_STUMP_REGRESSION_MODEL_ID               = 100340;
const int SERIALIZATION_STUMP_REGRESSION_TRAINING_RESULT_ID     = 100350;
const int SERIALIZATION_STUMP_REGRESSION_PREDICTION_RESULT_ID   = 100360;

const int SERIALIZATION_ADABOOST_MODEL_ID                      = 100400;
const int SERIALIZATION_ADABOOST_TRAINING_RESULT_ID            = 100410;
const int SERIALIZATION_MULTICLASS_ADABOOST_MODEL_ID           = 100420;
const int SERIALIZATION_MULTICLASS_ADABOOST_TRAINING_RESULT_ID = 100430;

const int SERIALIZATION_BROWNBOOST_MODEL_ID           = 100500;
const int SERIALIZATION_BROWNBOOST_TRAINING_RESULT_ID = 100510;

const int SERIALIZATION_LOGITBOOST_MODEL_ID           = 100600;
const int SERIALIZATION_LOGITBOOST_TRAINING_RESULT_ID = 100610;

const int SERIALIZATION_NAIVE_BAYES_MODEL_ID          = 100700;
const int SERIALIZATION_NAIVE_BAYES_PARTIALMODEL_ID   = 100710;
const int SERIALIZATION_NAIVE_BAYES_RESULT_ID         = 100720;
const int SERIALIZATION_NAIVE_BAYES_PARTIAL_RESULT_ID = 100730;

const int SERIALIZATION_SVM_MODEL_ID           = 100800;
const int SERIALIZATION_SVM_TRAINING_RESULT_ID = 100810;

const int SERIALIZATION_MULTI_CLASS_CLASSIFIER_MODEL_ID = 100900;
const int SERIALIZATION_MULTICLASS_CLASSIFIER_RESULT_ID = 100910;
const int SERIALIZATION_MULTICLASS_PREDICTION_RESULT_ID = 100920;

const int SERIALIZATION_COVARIANCE_PARTIAL_RESULT_ID = 101000;
const int SERIALIZATION_COVARIANCE_RESULT_ID         = 101010;

const int SERIALIZATION_KMEANS_PARTIAL_RESULT_ID                     = 101100;
const int SERIALIZATION_KMEANS_RESULT_ID                             = 101110;
const int SERIALIZATION_KMEANS_INIT_PARTIAL_RESULT_ID                = 101200;
const int SERIALIZATION_KMEANS_INIT_STEP2LOCAL_PP_PARTIAL_RESULT_ID  = 101210;
const int SERIALIZATION_KMEANS_INIT_STEP3MASTER_PP_PARTIAL_RESULT_ID = 101220;
const int SERIALIZATION_KMEANS_INIT_STEP4LOCAL_PP_PARTIAL_RESULT_ID  = 101230;
const int SERIALIZATION_KMEANS_INIT_STEP5MASTER_PP_PARTIAL_RESULT_ID = 101240;

const int SERIALIZATION_KMEANS_INIT_RESULT_ID = 101300;

const int SERIALIZATION_CLASSIFIER_TRAINING_PARTIAL_RESULT_ID            = 101400;
const int SERIALIZATION_CLASSIFIER_BINARY_CONFUSION_MATRIX_RESULT_ID     = 101410;
const int SERIALIZATION_CLASSIFIER_MULTICLASS_CONFUSION_MATRIX_RESULT_ID = 101420;
const int SERIALIZATION_CLASSIFIER_PREDICTION_RESULT_ID                  = 101430;
const int SERIALIZATION_CLASSIFIER_TRAINING_RESULT_ID                    = 101440;

const int SERIALIZATION_MOMENTS_PARTIAL_RESULT_ID = 101500;
const int SERIALIZATION_MOMENTS_RESULT_ID         = 101510;

const int SERIALIZATION_IMPLICIT_ALS_MODEL_ID                                          = 101600;
const int SERIALIZATION_IMPLICIT_ALS_PARTIALMODEL_ID                                   = 101610;
const int SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_PARTIAL_RESULT_ID              = 101620;
const int SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_RESULT_ID                      = 101630;
const int SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_RESULT_ID                           = 101640;
const int SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_PARTIAL_RESULT_BASE_ID              = 101645;
const int SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_PARTIAL_RESULT_ID                   = 101650;
const int SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_DISTRIBUTED_PARTIAL_RESULT_STEP2_ID = 101657;
const int SERIALIZATION_IMPLICIT_ALS_TRAINING_RESULT_ID                                = 101660;
const int SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP1_ID               = 101670;
const int SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP2_ID               = 101675;
const int SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID               = 101680;
const int SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP4_ID               = 101685;

const int SERIALIZATION_ASSOCIATION_RULES_RESULT_ID = 101700;

const int SERIALIZATION_CHOLESKY_RESULT_ID = 101800;

const int SERIALIZATION_CORRELATION_DISTANCE_RESULT_ID = 101900;
const int SERIALIZATION_COSINE_DISTANCE_RESULT_ID      = 101910;

const int SERIALIZATION_EM_GMM_INIT_RESULT_ID = 102000;
const int SERIALIZATION_EM_GMM_RESULT_ID      = 102010;

const int SERIALIZATION_KERNEL_FUNCTION_RESULT_ID = 102100;

const int SERIALIZATION_OUTLIER_DETECTION_MULTIVARIATE_RESULT_ID = 102200;
const int SERIALIZATION_OUTLIER_DETECTION_UNIVARIATE_RESULT_ID   = 102210;
const int SERIALIZATION_OUTLIER_DETECTION_BACON_RESULT_ID        = 102220;

const int SERIALIZATION_PIVOTED_QR_RESULT_ID = 102300;

const int SERIALIZATION_QR_RESULT_ID                           = 102400;
const int SERIALIZATION_QR_ONLINE_PARTIAL_RESULT_ID            = 102410;
const int SERIALIZATION_QR_DISTRIBUTED_PARTIAL_RESULT_ID       = 102420;
const int SERIALIZATION_QR_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID = 102430;

const int SERIALIZATION_QUANTILES_RESULT_ID = 102500;

const int SERIALIZATION_WEAK_LEARNER_RESULT_ID = 102600;

const int SERIALIZATION_SVD_RESULT_ID                           = 102700;
const int SERIALIZATION_SVD_ONLINE_PARTIAL_RESULT_ID            = 102710;
const int SERIALIZATION_SVD_DISTRIBUTED_PARTIAL_RESULT_ID       = 102720;
const int SERIALIZATION_SVD_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID = 102730;

const int SERIALIZATION_RELU_RESULT_ID = 103000;

const int SERIALIZATION_SORTING_RESULT_ID = 103100;

const int SERIALIZATION_SOFTMAX_RESULT_ID  = 103200;
const int SERIALIZATION_LOGISTIC_RESULT_ID = 103300;
const int SERIALIZATION_TANH_RESULT_ID     = 103400;

const int SERIALIZATION_SMOOTHRELU_RESULT_ID = 103500;
const int SERIALIZATION_ABS_RESULT_ID        = 103600;

const int SERIALIZATION_OBJECTIVE_FUNCTION_RESULT_ID = 103700;
const int SERIALIZATION_ITERATIVE_SOLVER_RESULT_ID   = 103810;
const int SERIALIZATION_ADAGRAD_RESULT_ID            = 103820;
const int SERIALIZATION_LBFGS_RESULT_ID              = 103830;
const int SERIALIZATION_SGD_RESULT_ID                = 103840;
const int SERIALIZATION_SAGA_RESULT_ID               = 103850;
const int SERIALIZATION_COORDINATE_DESCENT_RESULT_ID = 103860;

const int SERIALIZATION_NORMALIZATION_ZSCORE_RESULT_ID = 103900;
const int SERIALIZATION_NORMALIZATION_MINMAX_RESULT_ID = 103910;

const int SERIALIZATION_RIDGE_REGRESSION_MODELNORMEQ_ID       = 105000;
const int SERIALIZATION_RIDGE_REGRESSION_PARTIAL_RESULT_ID    = 105010;
const int SERIALIZATION_RIDGE_REGRESSION_TRAINING_RESULT_ID   = 105020;
const int SERIALIZATION_RIDGE_REGRESSION_PREDICTION_RESULT_ID = 105030;

const int SERIALIZATION_K_NEAREST_NEIGHBOR_MODEL_ID                = 106000;
const int SERIALIZATION_K_NEAREST_NEIGHBOR_BF_MODEL_ID             = 106001;
const int SERIALIZATION_K_NEAREST_NEIGHBOR_TRAINING_RESULT_ID      = 106010;
const int SERIALIZATION_K_NEAREST_NEIGHBOR_PREDICTION_RESULT_ID    = 106020;
const int SERIALIZATION_K_NEAREST_NEIGHBOR_BF_TRAINING_RESULT_ID   = 106030;
const int SERIALIZATION_K_NEAREST_NEIGHBOR_BF_PREDICTION_RESULT_ID = 106040;

const int SERIALIZATION_DECISION_FOREST_CLASSIFICATION_MODEL_ID             = 107000;
const int SERIALIZATION_DECISION_FOREST_CLASSIFICATION_TRAINING_RESULT_ID   = 107010;
const int SERIALIZATION_DECISION_FOREST_CLASSIFICATION_PREDICTION_RESULT_ID = 107020;
const int SERIALIZATION_DECISION_FOREST_REGRESSION_MODEL_ID                 = 107030;
const int SERIALIZATION_DECISION_FOREST_REGRESSION_TRAINING_RESULT_ID       = 107040;
const int SERIALIZATION_DECISION_FOREST_REGRESSION_PREDICTION_RESULT_ID     = 107050;
const int SERIALIZATION_GBT_CLASSIFICATION_MODEL_ID                         = 107100;
const int SERIALIZATION_GBT_CLASSIFICATION_TRAINING_RESULT_ID               = 107110;
const int SERIALIZATION_GBT_CLASSIFICATION_PREDICTION_RESULT_ID             = 107120;
const int SERIALIZATION_GBT_REGRESSION_MODEL_ID                             = 107130;
const int SERIALIZATION_GBT_REGRESSION_TRAINING_RESULT_ID                   = 107140;
const int SERIALIZATION_GBT_REGRESSION_PREDICTION_RESULT_ID                 = 107150;
const int SERIALIZATION_GBT_DECISION_TREE_ID                                = 107160;

const int SERIALIZATION_DECISION_TREE_CLASSIFICATION_MODEL_ID           = 108000;
const int SERIALIZATION_DECISION_TREE_CLASSIFICATION_TRAINING_RESULT_ID = 108010;
const int SERIALIZATION_DECISION_TREE_REGRESSION_MODEL_ID               = 108020;
const int SERIALIZATION_DECISION_TREE_REGRESSION_TRAINING_RESULT_ID     = 108030;
const int SERIALIZATION_DECISION_TREE_REGRESSION_PREDICTION_RESULT_ID   = 108040;

const int SERIALIZATION_REGRESSION_TRAINING_RESULT_ID   = 109000;
const int SERIALIZATION_REGRESSION_PREDICTION_RESULT_ID = 109020;

const int SERIALIZATION_LM_TRAINING_RESULT_ID   = 109100;
const int SERIALIZATION_LM_PREDICTION_RESULT_ID = 109120;

const int SERIALIZATION_LOGISTIC_REGRESSION_MODEL_ID             = 110000;
const int SERIALIZATION_LOGISTIC_REGRESSION_TRAINING_RESULT_ID   = 110010;
const int SERIALIZATION_LOGISTIC_REGRESSION_PREDICTION_RESULT_ID = 110020;

const int SERIALIZATION_DBSCAN_RESULT_ID                            = 120000;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP1_ID  = 120100;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP2_ID  = 120200;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID  = 120300;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP4_ID  = 120400;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP5_ID  = 120500;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP6_ID  = 120600;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP7_ID  = 120700;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP8_ID  = 120800;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP9_ID  = 120900;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_RESULT_STEP9_ID          = 120910;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP10_ID = 121000;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP11_ID = 121100;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP12_ID = 121200;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_RESULT_STEP13_ID         = 121300;
const int SERIALIZATION_DBSCAN_DISTRIBUTED_PARTIAL_RESULT_STEP13_ID = 121310;

} // namespace daal

#define DAAL_NEW_DELETE()                                          \
    static void * operator new(std::size_t sz)                     \
    {                                                              \
        return daal::services::daal_calloc(sz);                    \
    }                                                              \
    static void * operator new[](std::size_t sz)                   \
    {                                                              \
        return daal::services::daal_calloc(sz);                    \
    }                                                              \
    static void * operator new(std::size_t /*sz*/, void * where)   \
    {                                                              \
        return where;                                              \
    }                                                              \
    static void * operator new[](std::size_t /*sz*/, void * where) \
    {                                                              \
        return where;                                              \
    }                                                              \
    static void operator delete(void * ptr, std::size_t /*sz*/)    \
    {                                                              \
        daal::services::daal_free(ptr);                            \
    }                                                              \
    static void operator delete[](void * ptr, std::size_t /*sz*/)  \
    {                                                              \
        daal::services::daal_free(ptr);                            \
    }

#define DAAL_CAST_OPERATOR(ClassName)                                            \
    template <class U>                                                           \
    static services::SharedPtr<ClassName> cast(const services::SharedPtr<U> & r) \
    {                                                                            \
        return services::dynamicPointerCast<ClassName, U>(r);                    \
    }

#define DAAL_DOWN_CAST_OPERATOR(DstClassName, SrcClassName)                                        \
    static services::SharedPtr<DstClassName> downCast(const services::SharedPtr<SrcClassName> & r) \
    {                                                                                              \
        return services::dynamicPointerCast<DstClassName, SrcClassName>(r);                        \
    }

#ifndef DAAL_ALGORITHM_FP_TYPE
    #define DAAL_ALGORITHM_FP_TYPE float /* default type for algorithms */
#endif
#ifndef DAAL_DATA_TYPE
    #define DAAL_DATA_TYPE float /* default type for tables and tensors */
#endif
#ifndef DAAL_SUMMARY_STATISTICS_TYPE
    #define DAAL_SUMMARY_STATISTICS_TYPE float /* default type for summary statistics in data source */
#endif

#ifdef DEBUG_ASSERT
    #include <assert.h>
    #define DAAL_ASSERT(cond)     assert(cond);
    #define DAAL_ASSERT_DECL(var) var
#else
    #define DAAL_ASSERT(cond)
    #define DAAL_ASSERT_DECL(var)
#endif

#define DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_BOOL(type, op1, op2, result) \
    {                                                                      \
        result = false;                                                    \
        if (!(0 == (op1)) && !(0 == (op2)))                                \
        {                                                                  \
            volatile type r = (op1) * (op2);                               \
            r /= (op1);                                                    \
            if (!(r == (op2))) result = true;                              \
        }                                                                  \
    }

#define DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(type, op1, op2)                                     \
    {                                                                                             \
        if (!(0 == (op1)) && !(0 == (op2)))                                                       \
        {                                                                                         \
            volatile type r = (op1) * (op2);                                                      \
            r /= (op1);                                                                           \
            if (!(r == (op2))) return services::Status(services::ErrorBufferSizeIntegerOverflow); \
        }                                                                                         \
    }

#define DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_THROW_IF_POSSIBLE(type, op1, op2)                                       \
    {                                                                                                                 \
        if (!(0 == (op1)) && !(0 == (op2)))                                                                           \
        {                                                                                                             \
            volatile type r = (op1) * (op2);                                                                          \
            r /= (op1);                                                                                               \
            if (!(r == (op2))) services::throwIfPossible(services::Status(services::ErrorBufferSizeIntegerOverflow)); \
        }                                                                                                             \
    }

#define DAAL_OVERFLOW_CHECK_BY_ADDING(type, op1, op2)                                         \
    {                                                                                         \
        volatile type r = (op1) + (op2);                                                      \
        r -= (op1);                                                                           \
        if (!(r == (op2))) return services::Status(services::ErrorBufferSizeIntegerOverflow); \
    }

#define DAAL_OVERFLOW_CHECK_BY_ADDING_THROW_IF_POSSIBLE(type, op1, op2)                                           \
    {                                                                                                             \
        volatile type r = (op1) + (op2);                                                                          \
        r -= (op1);                                                                                               \
        if (!(r == (op2))) services::throwIfPossible(services::Status(services::ErrorBufferSizeIntegerOverflow)); \
    }

#define DAAL_CHECK_STATUS_RETURN_IF_FAIL(statVal, returnObj) \
    {                                                        \
        if (!(statVal)) return returnObj;                    \
    }

#define DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(statVal) \
    {                                                  \
        if (!(statVal)) return;                        \
    }

#define DAAL_CHECK(cond, error) \
    if (!(cond)) return services::Status(error);
#define DAAL_CHECK_EX(cond, error, detailType, detailValue) \
    if (!(cond)) return services::Status(services::Error::create(error, detailType, detailValue));
#define DAAL_CHECK_THR(cond, error) \
    {                               \
        using namespace services;   \
        if (!(cond))                \
        {                           \
            safeStat.add(error);    \
            return;                 \
        }                           \
    }

#define DAAL_CHECK_MALLOC(cond)     DAAL_CHECK(cond, services::ErrorMemoryAllocationFailed)
#define DAAL_CHECK_MALLOC_THR(cond) DAAL_CHECK_THR(cond, services::ErrorMemoryAllocationFailed)

#define DAAL_CHECK_STATUS(destVar, srcVal) \
    {                                      \
        destVar.add(srcVal);               \
        if (!(destVar)) return destVar;    \
    }
#define DAAL_CHECK_STATUS_VAR(statVal)  \
    {                                   \
        if (!(statVal)) return statVal; \
    }
#define DAAL_CHECK_STATUS_THR(statVal) \
    {                                  \
        if (!(statVal))                \
        {                              \
            safeStat.add(statVal);     \
            return;                    \
        }                              \
    }
#define DAAL_CHECK_SAFE_STATUS()                   \
    {                                              \
        if (!(safeStat)) return safeStat.detach(); \
    }
#define DAAL_CHECK_BREAK(cond) \
    {                          \
        if ((cond)) break;     \
    }
#define DAAL_CHECK_STATUS_OK(cond, status) \
    {                                      \
        if (!(cond)) return status;        \
    }
#define DAAL_CHECK_COND_ERROR(cond, status, error) \
    {                                              \
        if (!(cond)) (status).add(error);          \
    }

#define DAAL_CHECK_BLOCK_STATUS(block)                  \
    {                                                   \
        if (!(block).status()) return (block).status(); \
    }
#define DAAL_CHECK_BLOCK_STATUS_THR(block) DAAL_CHECK_STATUS_THR((block).status())

#define DAAL_DEFAULT_CREATE_IMPL(Type)                      \
    {                                                       \
        services::Status defaultSt;                         \
        services::Status & st = (stat ? *stat : defaultSt); \
        services::SharedPtr<Type> result(new Type(st));     \
        if (!result)                                        \
        {                                                   \
            st.add(services::ErrorMemoryAllocationFailed);  \
        }                                                   \
        if (!st)                                            \
        {                                                   \
            result.reset();                                 \
        }                                                   \
        return result;                                      \
    }

#define DAAL_DEFAULT_CREATE_IMPL_EX(Type, ...)                       \
    {                                                                \
        services::Status defaultSt;                                  \
        services::Status & st = (stat ? *stat : defaultSt);          \
        services::SharedPtr<Type> result(new Type(__VA_ARGS__, st)); \
        if (!result)                                                 \
        {                                                            \
            st.add(services::ErrorMemoryAllocationFailed);           \
        }                                                            \
        if (!st)                                                     \
        {                                                            \
            result.reset();                                          \
        }                                                            \
        return result;                                               \
    }

#define DAAL_TEMPLATE_ARGUMENTS(...) __VA_ARGS__

#define DAAL_DEFAULT_CREATE_TEMPLATE_IMPL(Type, TemplateArgs)                        \
    {                                                                                \
        services::Status defaultSt;                                                  \
        services::Status & st = (stat ? *stat : defaultSt);                          \
        services::SharedPtr<Type<TemplateArgs> > result(new Type<TemplateArgs>(st)); \
        if (!result)                                                                 \
        {                                                                            \
            st.add(services::ErrorMemoryAllocationFailed);                           \
        }                                                                            \
        if (!st)                                                                     \
        {                                                                            \
            result.reset();                                                          \
        }                                                                            \
        return result;                                                               \
    }

#define DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(Type, TemplateArgs, ...)                             \
    {                                                                                             \
        services::Status defaultSt;                                                               \
        services::Status & st = (stat ? *stat : defaultSt);                                       \
        services::SharedPtr<Type<TemplateArgs> > result(new Type<TemplateArgs>(__VA_ARGS__, st)); \
        if (!result)                                                                              \
        {                                                                                         \
            st.add(services::ErrorMemoryAllocationFailed);                                        \
        }                                                                                         \
        if (!st)                                                                                  \
        {                                                                                         \
            result.reset();                                                                       \
        }                                                                                         \
        return result;                                                                            \
    }

#define DAAL_CHECK_NUMERIC_TABLE(destVar, ...) DAAL_CHECK_STATUS(destVar, data_management::checkNumericTable(__VA_ARGS__))

#endif
