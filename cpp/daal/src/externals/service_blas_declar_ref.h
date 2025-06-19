/* file: service_blas_declar_ref.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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
//  Declarations for BLAS OpenBLAS functions.
//--
*/

#ifndef __SERVICE_BLAS_DECLAR_REF_H__
#define __SERVICE_BLAS_DECLAR_REF_H__

#include "services/daal_defines.h"

namespace daal
{
namespace internal
{
namespace ref
{
extern "C"
{
    extern void ssyrk_(const char *, const char *, const DAAL_INT *, const DAAL_INT *, const float *, const float *, const DAAL_INT *, const float *,
                       float *, const DAAL_INT *);
    extern void dsyrk_(const char *, const char *, const DAAL_INT *, const DAAL_INT *, const double *, const double *, const DAAL_INT *,
                       const double *, double *, const DAAL_INT *);

    extern void ssyr_(const char *, const DAAL_INT *, const float *, const float *, const DAAL_INT *, float *, const DAAL_INT *);
    extern void dsyr_(const char *, const DAAL_INT *, const double *, const double *, const DAAL_INT *, double *, const DAAL_INT *);

    extern void sgemm_(const char *, const char *, const DAAL_INT *, const DAAL_INT *, const DAAL_INT *, const float *, const float *,
                       const DAAL_INT *, const float *, const DAAL_INT *, const float *, const float *, const DAAL_INT *);
    extern void dgemm_(const char *, const char *, const DAAL_INT *, const DAAL_INT *, const DAAL_INT *, const double *, const double *,
                       const DAAL_INT *, const double *, const DAAL_INT *, const double *, const double *, const DAAL_INT *);

    extern void ssymm_(const char *, const char *, const DAAL_INT *, const DAAL_INT *, const float *, const float *, const DAAL_INT *, const float *,
                       const DAAL_INT *, const float *, float *, const DAAL_INT *);
    extern void dsymm_(const char *, const char *, const DAAL_INT *, const DAAL_INT *, const double *, const double *, const DAAL_INT *,
                       const double *, const DAAL_INT *, const double *, double *, const DAAL_INT *);

    extern void sgemv_(const char *, const DAAL_INT *, const DAAL_INT *, const float *, const float *, const DAAL_INT *, const float *,
                       const DAAL_INT *, const float *, const float *, const DAAL_INT *);
    extern void dgemv_(const char *, const DAAL_INT *, const DAAL_INT *, const double *, const double *, const DAAL_INT *, const double *,
                       const DAAL_INT *, const double *, const double *, const DAAL_INT *);

    extern void saxpy_(const DAAL_INT *, const float *, const float *, const DAAL_INT *, float *, const DAAL_INT *);
    extern void daxpy_(const DAAL_INT *, const double *, const double *, const DAAL_INT *, double *, const DAAL_INT *);

    extern float sdot_(const DAAL_INT *, const float *, const DAAL_INT *, const float *, const DAAL_INT *);
    extern double ddot_(const DAAL_INT *, const double *, const DAAL_INT *, const double *, const DAAL_INT *);

    extern void dscal_(const DAAL_INT *, const double *, double *, const DAAL_INT *);
    extern void sscal_(const DAAL_INT *, const float *, float *, const DAAL_INT *);
}

} // namespace ref
} // namespace internal
} // namespace daal

#endif
