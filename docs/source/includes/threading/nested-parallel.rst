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
   daal::tls<float *> tls([=, &safeStat]() {
      float * localBuffer = new (std::nothrow) float[localSize];
      if (!localBuffer) {
         safeStat.add(services::ErrorMemoryAllocationFailed);
      }
      return localBuffer;
   })
   daal::threader_for(n, n, [&](size_t i) {
      float * localBuffer = tls.local();
      if (!localBuffer) {
         // Allocation error happened earlier
         return;
      }

      // Initialize localBuffer with some data here

      daal::threader_for(m, m, [&](size_t j) {
         /* Some work */
      });

      // While executing the above parallel_for, the thread might have run iterations
      // of the outer parallel_for, and so might have changed the thread specific value.
      assert(localBuffer == tls.local()); // The assertion may fail!
   });
   DAAL_CHECK_SAFE_STATUS()

   tls.reduce([&](float * localBuffer) {
      if (localBuffer) {
         /* Do reduction */
         delete localBuffer;
      }
   });
