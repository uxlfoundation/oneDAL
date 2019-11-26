/* file: distributed_step3_local_input.cpp */
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

#include <jni.h>
#include "daal.h"
#include "com_intel_daal_algorithms_dbscan_DistributedStep3LocalInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::dbscan;

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep3LocalInput
* Method:    cSetDataCollection
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep3LocalInput_cSetDataCollection(JNIEnv *, jobject, jlong inputAddr,
                                                                                                           jint id, jlong dcAddr)
{
    if (id == partialData || id == partialWeights)
    { jniInput<DistributedInput<step3Local> >::set<LocalCollectionInputId, DataCollection>(inputAddr, id, dcAddr); }
    else
    {
        jniInput<DistributedInput<step3Local> >::set<Step3LocalCollectionInputId, DataCollection>(inputAddr, id, dcAddr);
    }
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep3LocalInput
* Method:    cAddNumericTable
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep3LocalInput_cAddNumericTable(JNIEnv *, jobject, jlong inputAddr, jint id,
                                                                                                         jlong ntAddr)
{
    if (id == partialData || id == partialWeights)
    { jniInput<DistributedInput<step3Local> >::add<LocalCollectionInputId, NumericTable>(inputAddr, id, ntAddr); }
    else
    {
        jniInput<DistributedInput<step3Local> >::add<Step3LocalCollectionInputId, NumericTable>(inputAddr, id, ntAddr);
    }
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep3LocalInput
* Method:    cGetDataCollection
* Signature:(JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep3LocalInput_cGetDataCollection(JNIEnv *, jobject, jlong inputAddr,
                                                                                                            jint id)
{
    if (id == partialData || id == partialWeights)
    { return jniInput<DistributedInput<step3Local> >::get<LocalCollectionInputId, DataCollection>(inputAddr, id); }
    else
    {
        return jniInput<DistributedInput<step3Local> >::get<Step3LocalCollectionInputId, DataCollection>(inputAddr, id);
    }
}
