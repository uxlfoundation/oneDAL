.. Copyright contributors to the oneDAL project
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

::

   /*
   //++
   //  Implementation of Abc training algorithm.
   //--
   */

   #include "src/algorithms/service_error_handling.h"
   #include "src/data_management/service_numeric_table.h"

   namespace daal::algorithms::abc::training::internal
   {

   /* Generic template implementation of cpuSpecificCode function for all data types
      and various instruction set architectures */
   template <typename algorithmFPType, CpuType cpu>
   services::Status cpuSpecificCode(/* arguments */)
   {
      /* Implementation */
   };

   #if defined(DAAL_INTEL_CPP_COMPILER) && (__CPUID__(DAAL_CPU) == __avx512__)

   /* Specialization of cpuSpecificCode function for double data type and Intel(R) AVX-512 instruction set */
   template <>
   services::Status cpuSpecificCode<double, avx512>(/* arguments */)
   {
      /* Implementation */
   };

   /* Specialization of cpuSpecificCode function for float data type and Intel(R) AVX-512 instruction set */
   template <>
   services::Status cpuSpecificCode<float, avx512>(/* arguments */)
   {
      /* Implementation */
   };

   #endif // DAAL_INTEL_CPP_COMPILER && (__CPUID__(DAAL_CPU) == __avx512__)

   template <typename algorithmFPType, CpuType cpu>
   services::Status AbcClassificationTrainingKernel<algorithmFPType, method2, cpu>::compute(/* arguments */)
   {
       services::Status status;

       /* Implementation that calls CPU-specific code: */
       status = cpuSpecificCode<algorithmFPType, cpu>(/* ... */);
       DAAL_CHECK_STATUS_VAR(status);

       /* Implementation continues */

       return status;
   }

   } // namespace daal::algorithms::abc::training::internal
