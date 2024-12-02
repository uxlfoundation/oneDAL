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

   #ifndef __ABC_CLASSIFICATION_TRAIN_KERNEL_H__
   #define __ABC_CLASSIFICATION_TRAIN_KERNEL_H__

   #include "src/algorithms/kernel.h"
   #include "data_management/data/numeric_table.h"    // NumericTable class
   /* Other necessary includes go here */

   using namespace daal::data_management;    // NumericTable class

   namespace daal::algorithms::abc::training::internal
   {
   /* Dummy base template class */
   template <typename algorithmFPType, Method method, CpuType cpu>
   class AbcClassificationTrainingKernel : public Kernel
   {};

   /* Computational kernel for 'method1' of the Abc training algoirthm */
   template <typename algorithmFPType, CpuType cpu>
   class AbcClassificationTrainingKernel<algorithmFPType, method1, cpu> : public Kernel
   {
   public:
      services::Status compute(/* Input and output arguments for the 'method1' */);
   };

   /* Computational kernel for 'method2' of the Abc training algoirthm */
   template <typename algorithmFPType, CpuType cpu>
   class AbcClassificationTrainingKernel<algorithmFPType, method2, cpu> : public Kernel
   {
   public:
      services::Status compute(/* Input and output arguments for the 'method2' */);
   };

   } // namespace daal::algorithms::abc::training::internal

   #endif // __ABC_CLASSIFICATION_TRAIN_KERNEL_H__
