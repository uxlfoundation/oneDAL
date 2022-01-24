/* file: explained_variance_batch.cpp */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
#include "daal.h"
#include "com_intel_daal_algorithms_pca_quality_metric_ExplainedVarianceBatch.h"
#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::pca::quality_metric;
using namespace daal::algorithms::pca::quality_metric::explained_variance;

/*
* Class:     com_intel_daal_algorithms_pca_quality_metric_ExplainedVarianceBatch
* Method:    cInit
* Signature: (II)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_quality_1metric_ExplainedVarianceBatch_cInit(JNIEnv *, jobject, jint prec, jint method)
{
    return jniBatch<explained_variance::Method, Batch, defaultDense>::newObj(prec, method);
}

/*
* Class:     com_intel_daal_algorithms_pca_quality_metric_ExplainedVarianceBatch
* Method:    cSetResult
* Signature: (JIIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_quality_1metric_ExplainedVarianceBatch_cSetResult(JNIEnv *, jobject, jlong algAddr,
                                                                                                            jint prec, jint method, jlong resultAddr)
{
    jniBatch<explained_variance::Method, Batch, defaultDense>::setResult<explained_variance::Result>(prec, method, algAddr, resultAddr);
}

/*
* Class:     com_intel_daal_algorithms_pca_quality_metric_ExplainedVarianceBatch
* Method:    cInitParameter
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_quality_1metric_ExplainedVarianceBatch_cInitParameter(JNIEnv *, jobject, jlong algAddr,
                                                                                                                 jint prec, jint method)
{
    return jniBatch<explained_variance::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_pca_quality_metric_ExplainedVarianceBatch
* Method:    cGetInput
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_quality_1metric_ExplainedVarianceBatch_cGetInput(JNIEnv *, jobject, jlong algAddr,
                                                                                                            jint prec, jint method)
{
    return jniBatch<explained_variance::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_pca_quality_metric_ExplainedVarianceBatch
* Method:    cGetResult
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_quality_1metric_ExplainedVarianceBatch_cGetResult(JNIEnv *, jobject, jlong algAddr,
                                                                                                             jint prec, jint method)
{
    return jniBatch<explained_variance::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_pca_quality_metric_ExplainedVarianceBatch
* Method:    cClone
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_quality_1metric_ExplainedVarianceBatch_cClone(JNIEnv *, jobject, jlong algAddr, jint prec,
                                                                                                         jint method)
{
    return jniBatch<explained_variance::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}
