.. ******************************************************************************
.. * Copyright contributors to the oneDAL project
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. highlight:: cpp

Threading Layer
^^^^^^^^^^^^^^^

|short_name| uses Intel\ |reg|\  oneAPI Threading Building Blocks (Intel\ |reg|\  oneTBB) to do parallel
computations on CPU. oneTBB is not used in the code of |short_name| algorithms directly. The algorithms rather
use custom primitives that either wrap oneTBB functionality or are in-house developed.
Those primitives form |short_name|'s threading layer.

This is done in order not to be dependent on possible oneTBB API changes and even
on the particular threading technology like oneTBB, C++11 standard threads, etc.

The API of the layer is defined in
`threading.h <https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/src/threading/threading.h>`_.
Please be aware that the threading API is not a part of |short_name| product API.
This is the product internal API that aimed to be used only by |short_name| developers, and can be changed at any time
without any prior notification.

This chapter describes common parallel patterns and primitives of the threading layer.

threader_for
************

Consider a case where you need to compute an elementwise sum of two arrays and store the results
into another array.
Here is a variant of sequential implementation:

.. include:: ../includes/threading/sum-sequential.rst

There are several options available in the threading layer of |short_name| to let the iterations of this code
run in parallel.
One of the options is to use ``daal::threader_for`` as shown here:

.. include:: ../includes/threading/sum-parallel.rst

The iteration space here goes from ``0`` to ``n-1``.
The last argument is a function object that performs a single iteration of the loop, given loop index ``i``.

Blocking
--------

To have more control over the parallel execution and to increase
`cache locality <https://en.wikipedia.org/wiki/Locality_of_reference>`_ |short_name| usually splits
the data into blocks and then processes those blocks in parallel.

This code shows how a typical parallel loop in |short_name| looks like:

.. include:: ../includes/threading/sum-parallel-by-blocks.rst

Thread-local Storage (TLS)
**************************

Consider you need to compute a dot product of two arrays.
Here is a variant of sequential implementation:

.. include:: ../includes/threading/dot-sequential.rst

Parallel computations can be performed in two steps:

    1. Compute partial dot product in each thread.
    2. Perform a reduction: Add the partial results from all threads to compute the final dot product.

``daal::tls`` provides a local storage where each thread can accumulate its local results.
The following code allocates memory that would store partial dot products for each thread:

.. include:: ../includes/threading/dot-parallel-init-tls.rst

``SafeStatus`` in this code denotes a thread-safe counterpart of the ``Status`` class.
``SafeStatus`` allows to collect errors from all threads and report them to the user using the
``detach()`` method. An example will be shown later in the documentation.

Checking the status right after the initialization code won't show the allocation errors,
because oneTBB uses lazy evaluation and the lambda function passed to the constructor of the TLS
is evaluated on first use of the thread-local storage (TLS).

There are several options available in the threading layer of |short_name| to compute the partial
dot product results at each thread.
One of the options is to use the already mentioned ``daal::threader_for`` and blocking approach
as shown here:

.. include:: ../includes/threading/dot-parallel-partial-compute.rst

To compute the final result it is required to reduce each thread's partial results
as shown here:

.. include:: ../includes/threading/dot-parallel-reduction.rst

Local memory of the threads should be released when it is no longer needed.

The complete parallel version of dot product computations would look like:

.. include:: ../includes/threading/dot-parallel.rst

Static Work Scheduling
**********************

By default, oneTBB uses
`dynamic work scheduling <https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/How_Task_Scheduler_Works.html>`_
and work stealing.
It means that two different runs of the same parallel loop can produce different
mappings of the loop's iteration space to the available threads.
This strategy is beneficial when it is difficult to estimate the amount of work performed
by each iteration.

In the cases when it is known that the iterations perform an equal amount of work, it
is more performant to use predefined mapping of the loop's iterations to threads.
This is what static work scheduling does.

``daal::static_threader_for`` and ``daal::static_tls`` allow implementation of static
work scheduling within |short_name|.

Here is a variant of parallel dot product computation with static scheduling:

.. include:: ../includes/threading/dot-static-parallel.rst

Nested Parallelism
******************

|short_name| supports nested parallel loops.
It is important to know that:

    "when a parallel construct calls another parallel construct, a thread can obtain a task
     from the outer-level construct while waiting for completion of the inner-level one."

    -- `oneTBB documentation <https://www.intel.com/content/www/us/en/docs/onetbb/developer-guide-api-reference/2021-13/work-isolation.html>`_

In practice, this means that a thread-local variable might unexpectedly
change its value after a nested parallel construct:

.. include:: ../includes/threading/nested-parallel.rst

In some scenarios this can lead to deadlocks, segmentation faults and other issues.

oneTBB provides ways to isolate execution of a parallel construct, for its tasks
to not interfere with other simultaneously running tasks.

Those options are preferred when the parallel loops are initially written as nested.
But in |short_name| there are cases when one parallel algorithm, the outer one,
calls another parallel algorithm, the inner one, within a parallel region.

The inner algorithm in this case can also be called solely, without additional nesting.
And we do not always want to make it isolated.

For the cases like that, |short_name| provides ``daal::ls``. Its ``local()`` method always
returns the same value for the same thread, regardless of the nested execution:

.. include:: ../includes/threading/nested-parallel-ls.rst
