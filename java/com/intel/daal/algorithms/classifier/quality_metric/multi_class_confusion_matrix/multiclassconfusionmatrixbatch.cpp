/* file: multiclassconfusionmatrixbatch.cpp */
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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
#include "daal.h"
#include "com_intel_daal_algorithms_classifier_quality_metric_multi_class_confusion_matrix_MultiClassConfusionMatrixBatch.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::classifier::quality_metric::multiclass_confusion_matrix;
using namespace daal::algorithms::classifier::quality_metric;

/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_multi_class_confusion_matrix_MultiClassConfusionMatrixBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL
    Java_com_intel_daal_algorithms_classifier_quality_1metric_multi_1class_1confusion_1matrix_MultiClassConfusionMatrixBatch_cInit(JNIEnv *, jobject,
                                                                                                                                   jint prec,
                                                                                                                                   jint method)
{
    return jniBatch<multiclass_confusion_matrix::Method, Batch, defaultDense>::newObj(prec, method);
}
/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_multi_class_confusion_matrix_MultiClassConfusionMatrixBatch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL
    Java_com_intel_daal_algorithms_classifier_quality_1metric_multi_1class_1confusion_1matrix_MultiClassConfusionMatrixBatch_cSetResult(
        JNIEnv *, jobject, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniBatch<multiclass_confusion_matrix::Method, Batch, defaultDense>::setResult<multiclass_confusion_matrix::Result>(prec, method, algAddr,
                                                                                                                       resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_multi_class_confusion_matrix_MultiClassConfusionMatrixBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL
    Java_com_intel_daal_algorithms_classifier_quality_1metric_multi_1class_1confusion_1matrix_MultiClassConfusionMatrixBatch_cInitParameter(
        JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multiclass_confusion_matrix::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_multi_class_confusion_matrix_MultiClassConfusionMatrixBatch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL
    Java_com_intel_daal_algorithms_classifier_quality_1metric_multi_1class_1confusion_1matrix_MultiClassConfusionMatrixBatch_cGetInput(
        JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multiclass_confusion_matrix::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_multi_class_confusion_matrix_MultiClassConfusionMatrixBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL
    Java_com_intel_daal_algorithms_classifier_quality_1metric_multi_1class_1confusion_1matrix_MultiClassConfusionMatrixBatch_cGetResult(
        JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multiclass_confusion_matrix::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_multi_class_confusion_matrix_MultiClassConfusionMatrixBatch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL
    Java_com_intel_daal_algorithms_classifier_quality_1metric_multi_1class_1confusion_1matrix_MultiClassConfusionMatrixBatch_cClone(JNIEnv *, jobject,
                                                                                                                                    jlong algAddr,
                                                                                                                                    jint prec,
                                                                                                                                    jint method)
{
    return jniBatch<multiclass_confusion_matrix::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}