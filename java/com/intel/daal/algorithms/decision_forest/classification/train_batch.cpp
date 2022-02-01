/* file: train_batch.cpp */
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
#include "com_intel_daal_algorithms_decision_forest_classification_training_TrainingBatch.h"
#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES()
namespace dfct = daal::algorithms::decision_forest::classification::training;

/*
* Class:     com_intel_daal_algorithms_decision_forest_classification_training_TrainingBatch
* Method:    cInit
* Signature: (IIJ)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1forest_classification_training_TrainingBatch_cInit(JNIEnv *, jobject thisObj,
                                                                                                                    jint prec, jint method,
                                                                                                                    jlong nClasses)
{
    return jniBatch<dfct::Method, dfct::Batch, dfct::defaultDense, dfct::hist>::newObj(prec, method, nClasses);
}

/*
* Class:     com_intel_daal_algorithms_decision_forest_classification_training_TrainingBatch
* Method:    cInitParameter
* Signature: (JIII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1forest_classification_training_TrainingBatch_cInitParameter(JNIEnv *,
                                                                                                                             jobject thisObj,
                                                                                                                             jlong algAddr, jint prec,
                                                                                                                             jint method, jint cmode)
{
    return jniBatch<dfct::Method, dfct::Batch, dfct::defaultDense, dfct::hist>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_decision_forest_classification_training_TrainingBatch
* Method:    cClone
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1forest_classification_training_TrainingBatch_cClone(JNIEnv *, jobject thisObj,
                                                                                                                     jlong algAddr, jint prec,
                                                                                                                     jint method)
{
    return jniBatch<dfct::Method, dfct::Batch, dfct::defaultDense, dfct::hist>::getClone(prec, method, algAddr);
}
