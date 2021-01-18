/* file: parameter.cpp */
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
#include <jni.h>/* Header for class com_intel_daal_algorithms_covariance_Offline */

#include "daal.h"
#include "com_intel_daal_algorithms_covariance_Parameter.h"

using namespace daal;
using namespace daal::algorithms;

#include "com_intel_daal_algorithms_covariance_OutputMatrixType.h"
#define CovMatrix com_intel_daal_algorithms_covariance_OutputMatrixType_CovarianceMatrix
#define CorMatrix com_intel_daal_algorithms_covariance_OutputMatrixType_CorrelationMatrix

/*
 * Class:     com_intel_daal_algorithms_covariance_Parameter
 * Method:    cSetDataSet
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_Parameter_cSetOutputDataType(JNIEnv * env, jobject thisObj, jlong parAddr, jint id)
{
    if (id == CovMatrix)
    {
        (*(covariance::Parameter *)parAddr).outputMatrixType = covariance::covarianceMatrix;
    }
    else if (id == CorMatrix)
    {
        (*(covariance::Parameter *)parAddr).outputMatrixType = covariance::correlationMatrix;
    }
}

JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_covariance_Parameter_cGetOutputDataType(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    return (jint)(*(covariance::Parameter *)parAddr).outputMatrixType;
}

/*
 * Class:     com_intel_daal_algorithms_covariance_Parameter
 * Method:    cSetCParameterObject
 * Signature: (JJIIII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_Parameter_cSetCParameterObject(JNIEnv * env, jobject thisObj, jlong parameterAddr,
                                                                                                jlong algAddr)
{
    using namespace daal::services;
    using namespace daal::algorithms;
    using namespace daal::data_management;

    SharedPtr<AlgorithmIface> * algShPtr = (SharedPtr<AlgorithmIface> *)algAddr;
    if (dynamicPointerCast<covariance::DistributedIface<step2Master>, AlgorithmIface>(*algShPtr))
    {
        staticPointerCast<covariance::DistributedIface<step2Master>, AlgorithmIface>(*algShPtr)->parameter =
            *((covariance::Parameter *)parameterAddr);
    }
    else
    {
        staticPointerCast<covariance::BatchImpl, AlgorithmIface>(*algShPtr)->parameter = *((covariance::Parameter *)parameterAddr);
    }
}
