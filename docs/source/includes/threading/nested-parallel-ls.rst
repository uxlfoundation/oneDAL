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

   #include "src/algorithms/service_error_handling.h"
   #include "src/threading/threading.h"

   SafeStatus safeStat;
   daal::ls<float *> ls([=, &safeStat]() {
      float * localBuffer = new (std::nothrow) float[localSize];
      if (!localBuffer) {
         safeStat.add(services::ErrorMemoryAllocationFailed);
      }
      return localBuffer;
   })
   daal::threader_for(n, n, [&](size_t i) {
      float * localBuffer = ls.local();
      if (!localBuffer) {
         // Allocation error happened earlier
         return;
      }

      // Initialize localBuffer with some data here

      daal::threader_for(m, m, [&](size_t j) {
         /* Some work */
      });

      // The thread specific value always stays unchanged after the nested execution.
      assert(localBuffer == ls.local()); // Assertion never fails!
   });
   DAAL_CHECK_SAFE_STATUS()

   ls.reduce([&](float * localBuffer) {
      if (localBuffer) {
         /* Do reduction */
         delete localBuffer;
      }
   });
