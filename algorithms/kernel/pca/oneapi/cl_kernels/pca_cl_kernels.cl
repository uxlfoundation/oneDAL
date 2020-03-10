/* file: pca_cl_kernels.cl */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

/*
//++
//  Implementation of PCA OpenCL kernels.
//--
*/

#ifndef __PCA_CL_KERNELS_CL__
#define __PCA_CL_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    pca_cl_kernels,

    __kernel void calculateVariances(__global algorithmFPType * covariance, __global algorithmFPType * variances) {
        const int tid       = get_global_id(0);
        const int nFeatures = get_global_size(0);

        variances[tid] = covariance[tid * nFeatures + tid];
    }

    __kernel void sortEigenvectorsDescending(__global const algorithmFPType * fullEigenvectors,
    __global const algorithmFPType * fullEigenvalues,
    __global algorithmFPType * eigenvectors,
    __global algorithmFPType * eigenvalues) {
        const int nFeatures = get_global_size(1);
        const int i = get_global_id(0);
        const int j = get_global_id(1);

        if (j == 0) {
            eigenvalues[i] = fullEigenvalues[nFeatures - 1 - i];
        }
        eigenvectors[i * nFeatures + j] = fullEigenvectors[(nFeatures - 1 - i) * nFeatures + j];
    }

);

#endif
