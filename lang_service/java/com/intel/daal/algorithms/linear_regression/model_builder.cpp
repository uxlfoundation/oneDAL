/* file: model_builder.cpp */
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
#include "com_intel_daal_algorithms_linear_regression_ModelBuilder.h"
#include "common_helpers_functions.h"

using namespace daal;
using namespace daal::algorithms::linear_regression;
using namespace daal::data_management;
using namespace daal::services;

/*
* Class:     com_intel_daal_algorithms_linear_regression_ModelBuilder
* Method:    cInit
* Signature: (JIII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_ModelBuilder_cInit(JNIEnv *, jobject, jint prec, jlong nFeatures,
                                                                                             jlong nResponses)
{
    if (prec == 0) { return (jlong)(new SharedPtr<ModelBuilder<double> >(new ModelBuilder<double>(nFeatures, nResponses))); }
    else
    {
        return (jlong)(new SharedPtr<ModelBuilder<float> >(new ModelBuilder<float>(nFeatures, nResponses)));
    }
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_ModelBuilder
 * Method:    cGetModel
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_ModelBuilder_cGetModel(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                 jint prec)
{
    ModelPtr * model = new ModelPtr;
    if (prec == 0)
    {
        services::SharedPtr<ModelBuilder<double> > * ptr = new services::SharedPtr<ModelBuilder<double> >();
        *ptr                                             = staticPointerCast<ModelBuilder<double> >(*(SharedPtr<ModelBuilder<double> > *)algAddr);
        *model                                           = staticPointerCast<Model>((*ptr)->getModel());
    }
    else
    {
        services::SharedPtr<ModelBuilder<float> > * ptr = new services::SharedPtr<ModelBuilder<float> >();
        *ptr                                            = staticPointerCast<ModelBuilder<float> >(*(SharedPtr<ModelBuilder<float> > *)algAddr);
        *model                                          = staticPointerCast<Model>((*ptr)->getModel());
    }
    return (jlong)model;
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_ModelBuilder
 * Method:    cSetBetaFloat
 * Signature:(JII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_linear_1regression_ModelBuilder_cSetBetaFloat(JNIEnv * env, jobject, jlong algAddr,
                                                                                                    jobject byteBuffer, jlong nBetas)
{
    float * firstBeta                               = (float *)(env->GetDirectBufferAddress(byteBuffer));
    float * lastBeta                                = firstBeta + nBetas;
    services::SharedPtr<ModelBuilder<float> > * ptr = new services::SharedPtr<ModelBuilder<float> >();
    *ptr                                            = staticPointerCast<ModelBuilder<float> >(*(SharedPtr<ModelBuilder<float> > *)algAddr);
    (*ptr)->setBeta(firstBeta, lastBeta);
    DAAL_CHECK_THROW((*ptr)->getStatus());
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_ModelBuilder
 * Method:    cSetBetaDouble
 * Signature:(JII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_linear_1regression_ModelBuilder_cSetBetaDouble(JNIEnv * env, jobject, jlong algAddr,
                                                                                                     jobject byteBuffer, jlong nBetas)
{
    double * firstBeta                               = (double *)(env->GetDirectBufferAddress(byteBuffer));
    double * lastBeta                                = firstBeta + nBetas;
    services::SharedPtr<ModelBuilder<double> > * ptr = new services::SharedPtr<ModelBuilder<double> >();
    *ptr                                             = staticPointerCast<ModelBuilder<double> >(*(SharedPtr<ModelBuilder<double> > *)algAddr);
    (*ptr)->setBeta(firstBeta, lastBeta);
    DAAL_CHECK_THROW((*ptr)->getStatus());
}
