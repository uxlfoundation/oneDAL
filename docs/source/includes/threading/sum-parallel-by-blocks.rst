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

   #include "src/threading/threading.h"

   void sum(const size_t n, const float* a, const float* b, float* c) {
      constexpr size_t blockSize = 256;
      const size_t nBlocks = (n + blockSize - 1) / blockSize;

      daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
         const size_t iStart = iBlock * blockSize;
         const size_t iEnd = (iBlock < (nBlocks - 1)) ? iStart + blockSize : n;
         for (size_t i = iStart; i < iEnd; ++i) {
            c[i] = a[i] + b[i];
         }
      });
   }
