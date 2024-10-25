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

   constexpr size_t blockSize = 1024;
   const size_t nBlocks = (n + blockSize - 1) / blockSize;

   daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
      const size_t iStart = iBlock * blockSize;
      const size_t iEnd = (iBlock < (nBlocks - 1)) ? iStart + blockSize : n;

      // Compute partial result for this block
      float partialDotProduct = 0.0f;
      for (size_t i = iStart; i < iEnd; ++i) {
         partialDotProduct += a[i] * b[i];
      }

      // Update thread-local result
      float * localDotProduct = dotProductTLS.local();
      if (!localDotProduct) {
         // Allocation error happened earlier
         return;
      }
      localDotProduct[0] += partialDotProduct;
   });
   DAAL_CHECK_SAFE_STATUS();  // if (!safeStat) return safeStat.detach();
