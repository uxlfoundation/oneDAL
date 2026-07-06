/* file: service_qsort.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Implementation of sorting algorithms.
//--
*/

#ifndef __SERVICE_QSORT_H__
#define __SERVICE_QSORT_H__

namespace daal
{
namespace algorithms
{
namespace internal
{
template <typename T>
inline void swap(T & x, T & y)
{
    T tmp = x;
    x     = y;
    y     = tmp;
}

template <typename algorithmDataType>
void qSort(size_t n, algorithmDataType * x)
{
    int i, ir, j, k, jstack = -1, l = 0;
    algorithmDataType a;
    const int M = 7, NSTACK = 128;
    int istack[NSTACK];

    ir = n - 1;

    for (;;)
    {
        if (ir - l < M)
        {
            for (j = l + 1; j <= ir; j++)
            {
                a = x[j];

                for (i = j - 1; i >= l; i--)
                {
                    if (x[i] <= a)
                    {
                        break;
                    }
                    x[i + 1] = x[i];
                }

                x[i + 1] = a;
            }

            if (jstack < 0)
            {
                break;
            }

            ir = istack[jstack--];
            l  = istack[jstack--];
        }
        else
        {
            k = (l + ir) >> 1;
            swap<algorithmDataType>(x[k], x[l + 1]);
            if (x[l] > x[ir])
            {
                swap<algorithmDataType>(x[l], x[ir]);
            }
            if (x[l + 1] > x[ir])
            {
                swap<algorithmDataType>(x[l + 1], x[ir]);
            }
            if (x[l] > x[l + 1])
            {
                swap<algorithmDataType>(x[l], x[l + 1]);
            }
            i = l + 1;
            j = ir;
            a = x[l + 1];
            for (;;)
            {
                while (x[++i] < a);
                while (x[--j] > a);
                if (j < i)
                {
                    break;
                }
                swap<algorithmDataType>(x[i], x[j]);
            }
            x[l + 1] = x[j];

            x[j] = a;
            jstack += 2;

            if (ir - i + 1 >= j - l)
            {
                istack[jstack]     = ir;
                istack[jstack - 1] = i;
                ir                 = j - 1;
            }
            else
            {
                istack[jstack]     = j - 1;
                istack[jstack - 1] = l;
                l                  = i;
            }
        }
    }
}
} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
