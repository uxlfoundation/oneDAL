/* file: binaryconfusionmatrixinput.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "com_intel_daal_algorithms_classifier_quality_metric_binary_confusion_matrix_BinaryConfusionMatrixInput.h"

#include "common_helpers.h"

#include "com_intel_daal_algorithms_classifier_quality_metric_binary_confusion_matrix_BinaryConfusionMatrixInputId.h"
#define PredictedLabels   com_intel_daal_algorithms_classifier_quality_metric_binary_confusion_matrix_BinaryConfusionMatrixInputId_PredictedLabels
#define GroundTruthLabels com_intel_daal_algorithms_classifier_quality_metric_binary_confusion_matrix_BinaryConfusionMatrixInputId_GroundTruthLabels

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::classifier::quality_metric;
using namespace daal::algorithms::classifier::quality_metric::binary_confusion_matrix;

/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_binary_confusion_matrix_BinaryConfusionMatrixInput
 * Method:    cSetInputTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_classifier_quality_1metric_binary_1confusion_1matrix_BinaryConfusionMatrixInput_cSetInputTable(
    JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id != PredictedLabels && id != GroundTruthLabels) return;

    jniInput<binary_confusion_matrix::Input>::set<binary_confusion_matrix::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_binary_confusion_matrix_BinaryConfusionMatrixInput
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_classifier_quality_1metric_binary_1confusion_1matrix_BinaryConfusionMatrixInput_cGetInputTable(
    JNIEnv *, jobject, jlong inputAddr, jint id)
{
    if (id != PredictedLabels && id != GroundTruthLabels) return (jlong)-1;

    return jniInput<binary_confusion_matrix::Input>::get<binary_confusion_matrix::InputId, NumericTable>(inputAddr, id);
}
