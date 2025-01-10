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
   //  instantiations of method1 of the Abc training algorithm.
   //--
   */

   #include "src/algorithms/abc/abc_classification_train_kernel.h"
   #include "src/algorithms/abc/abc_classification_train_method1_impl.i"

   namespace daal::algorithms::abc::training::internal
   {
   template class DAAL_EXPORT AbcClassificationTrainingKernel<DAAL_FPTYPE, method1, DAAL_CPU>;
   } // namespace daal::algorithms::abc::training::internal
