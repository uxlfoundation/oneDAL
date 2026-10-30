/* file: service_blas_declar_mkl.h */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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
//  Declarations of the MKL Fortran BLAS-like functions that MKL C headers do not declare.
//--
*/

#ifndef __SERVICE_BLAS_DECLAR_MKL_H__
#define __SERVICE_BLAS_DECLAR_MKL_H__

#include <mkl_types.h>

namespace daal
{
namespace internal
{
namespace mkl
{
extern "C"
{
    extern void mkl_somatcopy_(const char *, const char *, const MKL_INT *, const MKL_INT *, const float *, const float *, const MKL_INT *, float *,
                               const MKL_INT *);
    extern void mkl_domatcopy_(const char *, const char *, const MKL_INT *, const MKL_INT *, const double *, const double *, const MKL_INT *,
                               double *, const MKL_INT *);

    extern void mkl_somatadd_(const char *, const char *, const char *, const MKL_INT *, const MKL_INT *, const float *, const float *,
                              const MKL_INT *, const float *, const float *, const MKL_INT *, float *, const MKL_INT *);
    extern void mkl_domatadd_(const char *, const char *, const char *, const MKL_INT *, const MKL_INT *, const double *, const double *,
                              const MKL_INT *, const double *, const double *, const MKL_INT *, double *, const MKL_INT *);
}

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
