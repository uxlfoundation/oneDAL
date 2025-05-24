.. Copyright 2019 Intel Corporation
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

Managing Memory
===============

To improve performance of your application that calls |short_name|,
align your arrays on 64-byte boundaries and ensure that the leading
dimensions of the arrays are divisible by 64. For that purpose |short_name|
provides ``daal_malloc()`` and ``daal_free()`` functions to allocate and
deallocate memory.

To allocate memory, ``call daal_malloc()`` with the specified size of the
buffer to be allocated and the alignment of the buffer, which must be
a power of 2. If the specified alignment is not a power of 2, the
library uses the 32-byte alignment.

To deallocate memory allocated earlier by the ``daal_malloc()`` function,
call the ``daal_free()`` function and set a pointer to the buffer to be freed.
