/* file: service_lapack_declar_ref.h */
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
//  Declarations for Lapack OpenBLAS functions.
//--
*/

#ifndef __SERVICE_LAPACK_DECLAR_REF_H__
#define __SERVICE_LAPACK_DECLAR_REF_H__

#include "services/daal_defines.h"

namespace daal
{
namespace internal
{
namespace ref
{
extern "C"
{
    extern void sgetrf_(DAAL_INT *, DAAL_INT *, float *, DAAL_INT *, DAAL_INT *, DAAL_INT *);
    extern void dgetrf_(DAAL_INT *, DAAL_INT *, double *, DAAL_INT *, DAAL_INT *, DAAL_INT *);

    extern void sgetrs_(char *, DAAL_INT *, DAAL_INT *, float *, DAAL_INT *, DAAL_INT *, float *, DAAL_INT *, DAAL_INT *);
    extern void dgetrs_(char *, DAAL_INT *, DAAL_INT *, double *, DAAL_INT *, DAAL_INT *, double *, DAAL_INT *, DAAL_INT *);

    extern void spotrf_(char *, DAAL_INT *, float *, DAAL_INT *, DAAL_INT *);
    extern void dpotrf_(char *, DAAL_INT *, double *, DAAL_INT *, DAAL_INT *);

    extern void spotrs_(char *, DAAL_INT *, DAAL_INT *, float *, DAAL_INT *, float *, DAAL_INT *, DAAL_INT *);
    extern void dpotrs_(char *, DAAL_INT *, DAAL_INT *, double *, DAAL_INT *, double *, DAAL_INT *, DAAL_INT *);

    extern void spotri_(char *, DAAL_INT *, float *, DAAL_INT *, DAAL_INT *);
    extern void dpotri_(char *, DAAL_INT *, double *, DAAL_INT *, DAAL_INT *);

    extern void sgerqf_(DAAL_INT *, DAAL_INT *, float *, DAAL_INT *, float *, float *, DAAL_INT *, DAAL_INT *);
    extern void dgerqf_(DAAL_INT *, DAAL_INT *, double *, DAAL_INT *, double *, double *, DAAL_INT *, DAAL_INT *);

    extern void sormrq_(char *, char *, DAAL_INT *, DAAL_INT *, DAAL_INT *, float *, DAAL_INT *, float *, float *, DAAL_INT *, float *, DAAL_INT *,
                        DAAL_INT *);
    extern void dormrq_(char *, char *, DAAL_INT *, DAAL_INT *, DAAL_INT *, double *, DAAL_INT *, double *, double *, DAAL_INT *, double *,
                        DAAL_INT *, DAAL_INT *);

    extern void strtrs_(char *, char *, char *, DAAL_INT *, DAAL_INT *, float *, DAAL_INT *, float *, DAAL_INT *, DAAL_INT *);
    extern void dtrtrs_(char *, char *, char *, DAAL_INT *, DAAL_INT *, double *, DAAL_INT *, double *, DAAL_INT *, DAAL_INT *);

    extern void spptrf_(char *, DAAL_INT *, float *, DAAL_INT *);
    extern void dpptrf_(char *, DAAL_INT *, double *, DAAL_INT *);

    extern void sgeqrf_(DAAL_INT *, DAAL_INT *, float *, DAAL_INT *, float *, float *, DAAL_INT *, DAAL_INT *);
    extern void dgeqrf_(DAAL_INT *, DAAL_INT *, double *, DAAL_INT *, double *, double *, DAAL_INT *, DAAL_INT *);

    extern void sgeqp3_(const DAAL_INT *, const DAAL_INT *, float *, const DAAL_INT *, DAAL_INT *, float *, float *, const DAAL_INT *, DAAL_INT *);
    extern void dgeqp3_(const DAAL_INT *, const DAAL_INT *, double *, const DAAL_INT *, DAAL_INT *, double *, double *, const DAAL_INT *, DAAL_INT *);

    extern void sorgqr_(const DAAL_INT *, const DAAL_INT *, const DAAL_INT *, float *, const DAAL_INT *, const float *, float *, const DAAL_INT *,
                        DAAL_INT *);
    extern void dorgqr_(const DAAL_INT *, const DAAL_INT *, const DAAL_INT *, double *, const DAAL_INT *, const double *, double *, const DAAL_INT *,
                        DAAL_INT *);

    extern void sgesvd_(char *, char *, DAAL_INT *, DAAL_INT *, float *, DAAL_INT *, float *, float *, DAAL_INT *, float *, DAAL_INT *, float *,
                        DAAL_INT *, DAAL_INT *);
    extern void dgesvd_(char *, char *, DAAL_INT *, DAAL_INT *, double *, DAAL_INT *, double *, double *, DAAL_INT *, double *, DAAL_INT *, double *,
                        DAAL_INT *, DAAL_INT *);

    extern void ssyevd_(char *, char *, DAAL_INT *, float *, DAAL_INT *, float *, float *, DAAL_INT *, DAAL_INT *, DAAL_INT *, DAAL_INT *);
    extern void dsyevd_(char *, char *, DAAL_INT *, double *, DAAL_INT *, double *, double *, DAAL_INT *, DAAL_INT *, DAAL_INT *, DAAL_INT *);

    extern void ssyevr_(const char *, const char *, const char *, const DAAL_INT *, float *, const DAAL_INT *, const float *, const float *,
                        const DAAL_INT *, const DAAL_INT *, const float *, DAAL_INT *, float *, float *, const DAAL_INT *, DAAL_INT *, float *,
                        const DAAL_INT *, DAAL_INT *, const DAAL_INT *, DAAL_INT *);
    extern void dsyevr_(const char *, const char *, const char *, const DAAL_INT *, double *, const DAAL_INT *, const double *, const double *,
                        const DAAL_INT *, const DAAL_INT *, const double *, DAAL_INT *, double *, double *, const DAAL_INT *, DAAL_INT *, double *,
                        const DAAL_INT *, DAAL_INT *, const DAAL_INT *, DAAL_INT *);

    extern void sormqr_(char *, char *, DAAL_INT *, DAAL_INT *, DAAL_INT *, float *, DAAL_INT *, float *, float *, DAAL_INT *, float *, DAAL_INT *,
                        DAAL_INT *);
    extern void dormqr_(char *, char *, DAAL_INT *, DAAL_INT *, DAAL_INT *, double *, DAAL_INT *, double *, double *, DAAL_INT *, double *,
                        DAAL_INT *, DAAL_INT *);

    extern void drscl_(const DAAL_INT *, const double *, double *, const DAAL_INT *);
    extern void srscl_(const DAAL_INT *, const float *, float *, const DAAL_INT *);
}

} // namespace ref
} // namespace internal
} // namespace daal

#endif
