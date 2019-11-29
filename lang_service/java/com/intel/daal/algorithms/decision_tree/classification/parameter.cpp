/* file: parameter.cpp */
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
#include "com_intel_daal_algorithms_decision_tree_classification_Parameter.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()

namespace dt  = decision_tree;
namespace dtc = decision_tree::classification;

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_Parameter
* Method:    cGetPruning
* Signature: (J)I
*/
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_Parameter_cGetPruning(JNIEnv * env, jobject thisObj,
                                                                                                          jlong parAddr)
{
    return (jlong)(*(dtc::Parameter *)parAddr).pruning;
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_Parameter
* Method:    cSetPruning
* Signature: (JI)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_Parameter_cSetPruning(JNIEnv * env, jobject thisObj,
                                                                                                          jlong parAddr, jint value)
{
    (*(dtc::Parameter *)parAddr).pruning = (dt::Pruning)value;
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_Parameter
* Method:    cGetMaxTreeDepth
* Signature: (J)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_Parameter_cGetMaxTreeDepth(JNIEnv * env, jobject thisObj,
                                                                                                                jlong parAddr)
{
    return (jlong)(*(dtc::Parameter *)parAddr).maxTreeDepth;
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_Parameter
* Method:    cSetMaxTreeDepth
* Signature: (JJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_Parameter_cSetMaxTreeDepth(JNIEnv * env, jobject thisObj,
                                                                                                               jlong parAddr, jlong value)
{
    (*(dtc::Parameter *)parAddr).maxTreeDepth = (size_t)value;
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_Parameter
* Method:    cGetMinObservationsInLeafNodes
* Signature: (J)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_Parameter_cGetMinObservationsInLeafNodes(JNIEnv * env,
                                                                                                                              jobject thisObj,
                                                                                                                              jlong parAddr)
{
    return (jlong)(*(dtc::Parameter *)parAddr).minObservationsInLeafNodes;
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_Parameter
* Method:    cSetMinObservationsInLeafNodes
* Signature: (JJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_Parameter_cSetMinObservationsInLeafNodes(JNIEnv * env,
                                                                                                                             jobject thisObj,
                                                                                                                             jlong parAddr,
                                                                                                                             jlong value)
{
    (*(dtc::Parameter *)parAddr).minObservationsInLeafNodes = (size_t)value;
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_Parameter
* Method:    cGetSplitCriterion
* Signature: (J)I
*/
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_Parameter_cGetSplitCriterion(JNIEnv * env, jobject thisObj,
                                                                                                                 jlong parAddr)
{
    return (jlong)(*(dtc::Parameter *)parAddr).splitCriterion;
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_Parameter
* Method:    cSetSplitCriterion
* Signature: (JI)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_Parameter_cSetSplitCriterion(JNIEnv * env, jobject thisObj,
                                                                                                                 jlong parAddr, jint value)
{
    (*(dtc::Parameter *)parAddr).splitCriterion = (dtc::SplitCriterion)value;
}
