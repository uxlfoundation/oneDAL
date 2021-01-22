/* file: train_result.cpp */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
#include "com_intel_daal_algorithms_decision_tree_regression_training_TrainingResult.h"
#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES()

namespace dtr  = daal::algorithms::decision_tree::regression;
namespace dtrt = daal::algorithms::decision_tree::regression::training;

#include "com_intel_daal_algorithms_decision_tree_regression_training_TrainingResultId.h"
#define modelId com_intel_daal_algorithms_decision_tree_regression_training_TrainingResultId_modelId

/*
* Class:     com_intel_daal_algorithms_decision_tree_regression_training_TrainingResult
* Method:    cGetModel
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_regression_training_TrainingResult_cGetModel(JNIEnv * env, jobject thisObj,
                                                                                                                   jlong resAddr, jint id)
{
    if (id == modelId)
    {
        return jniArgument<dtrt::Result>::get<dtrt::ResultId, dtr::Model>(resAddr, dtrt::model);
    }

    return (jlong)0;
}
