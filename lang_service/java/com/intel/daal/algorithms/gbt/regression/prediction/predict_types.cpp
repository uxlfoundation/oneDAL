/* file: predict_types.cpp */
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
#include "com_intel_daal_algorithms_gbt_regression_prediction_PredictionInput.h"
#include "com_intel_daal_algorithms_gbt_regression_prediction_PredictionResult.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
namespace dfrp = daal::algorithms::gbt::regression::prediction;

/*
* Class:     com_intel_daal_algorithms_gbt_regression_prediction_PredictionInput
* Method:    cSetInputTable
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_gbt_regression_prediction_PredictionInput_cSetInputTable(JNIEnv * env, jobject thisObj,
                                                                                                               jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<dfrp::Input>::set<dfrp::NumericTableInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_prediction_PredictionInput
* Method:    cGetInputTable
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_prediction_PredictionInput_cGetInputTable(JNIEnv * env, jobject thisObj,
                                                                                                                jlong inputAddr, jint id)
{
    if (id != dfrp::data) return (jlong)-1;

    return jniInput<dfrp::Input>::get<dfrp::NumericTableInputId, NumericTable>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_prediction_PredictionInput
* Method:    cSetInputModel
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_gbt_regression_prediction_PredictionInput_cSetInputModel(JNIEnv * env, jobject thisObj,
                                                                                                               jlong inputAddr, jint id, jlong ntAddr)
{
    if (id != dfrp::model) return;

    jniInput<dfrp::Input>::set<dfrp::ModelInputId, gbt::regression::Model>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_prediction_PredictionInput
* Method:    cGetInputModel
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_prediction_PredictionInput_cGetInputModel(JNIEnv * env, jobject thisObj,
                                                                                                                jlong inputAddr, jint id)
{
    if (id != dfrp::model) return (jlong)-1;

    return jniInput<dfrp::Input>::get<dfrp::ModelInputId, gbt::regression::Model>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_prediction_PredictionInput
* Method:    cInit
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_prediction_PredictionInput_cInit(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                       jint prec, jint method)
{
    return jniBatch<dfrp::Method, dfrp::Batch, dfrp::defaultDense>::getInput(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_prediction_PredictionResult
* Method:    cNewResult
* Signature: ()J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_prediction_PredictionResult_cNewResult(JNIEnv *, jobject)
{
    return jniArgument<dfrp::Result>::newObj();
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_prediction_PredictionResult
* Method:    cGetResult
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_prediction_PredictionResult_cGetResult(JNIEnv * env, jobject thisObj,
                                                                                                             jlong algAddr, jint prec, jint method)
{
    return jniBatch<dfrp::Method, dfrp::Batch, dfrp::defaultDense>::getResult(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_prediction_PredictionResult
* Method:    cGetResultTable
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_prediction_PredictionResult_cGetResultTable(JNIEnv * env, jobject thisObj,
                                                                                                                  jlong resAddr, jint id)
{
    return jniArgument<dfrp::Result>::get<dfrp::ResultId, NumericTable>(resAddr, dfrp::ResultId(id));
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_prediction_PredictionResult
* Method:    cSetResultTable
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_gbt_regression_prediction_PredictionResult_cSetResultTable(JNIEnv *, jobject, jlong resAddr,
                                                                                                                 jint id, jlong numTableAddr)
{
    jniArgument<dfrp::Result>::set<dfrp::ResultId, NumericTable>(resAddr, dfrp::ResultId(id), numTableAddr);
}
