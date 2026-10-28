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

Using oneDAL Verbose Mode
=========================

When building applications that call Intel® oneAPI Data Analytics Library functions, it may be useful to determine:

- Which computational kernels are called
- What parameters are passed to them
- How much time is spent to execute the functions
- Which device (CPU/GPU) the kernel is executed on

You can get an application to print this information to a standard output
device by enabling **Intel® oneAPI Data Analytics Library Verbose Mode**.
All logs are printed to the `stderr` stream.

When Verbose mode is active in oneDAL, every call of a verbose-enabled function finishes with
printing a human-readable line describing the call. Even if the application gets terminated during
the function call, information for that function will be printed.
The first call to a verbose-enabled function also prints a version information line.
The analyzer prints its report when the process exits, or earlier on request (see
`Reports on request`_).

All libraries of the process (``libonedal_core``, ``libonedal``, ``libonedal_dpc``) record into one
tree, so a DAAL kernel called from a oneAPI algorithm is nested under that algorithm's task.

For GPU applications, additional information (one or more GPU information lines) will also 
be printed on the first call to a verbose-enabled function, following the version information lines printed
for the host CPU.

Verbosity Levels
----------------

We have common implementations for verbose mode with CPU and GPU applications.

 Levels:
  - Disabled (default)
  - Logger enabled
  - Tracer enabled
  - Analyzer enabled
  - All tools enabled
  - All tools and service functions enabled

Setting Verbose Mode
--------------------

You can change the verbose mode either by setting the environment variable ``ONEDAL_VERBOSE``.

**Using environment variable:**

+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=0`` | Disabled (default)                            |
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=1`` | Logger enabled                                |
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=2`` | Tracer enabled                                |
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=3`` | Analyzer enabled                              |
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=4`` | All tools enabled (Logger + Tracer + Analyzer)|
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=5`` | All tools and service functions enabled       |
+--------------------------+-----------------------------------------------+


A task given a SYCL queue waits for the queue when it starts and when it ends, so its time covers
the device work submitted inside it and nothing submitted before it. Profiling therefore changes
how device work overlaps; time a GPU workload without ``ONEDAL_VERBOSE`` for throughput figures.

Logger
------

**Logger** is a subtool used to capture information about each launched kernel.

Example output (for ``cov_dense_batch``):

::

    auto oneapi::dal::covariance::backend::compute_kernel_dense_impl<float>::operator()(const descriptor_t &, const parameters_t &, const input_t &)::(anonymous class)::operator()() const [Float = float]
    Profiler task_name: allreduce_sums, q_

    auto oneapi::dal::covariance::backend::compute_kernel_dense_impl<float>::operator()(const descriptor_t &, const parameters_t &, const input_t &)::(anonymous class)::operator()() const [Float = float]
    Profiler task_name: gemm, q_

Each entry shows the file and line of the ``PROFILER_TASK``, its name and its arguments; level 5
adds the ``__PRETTY_FUNCTION__``. Each call site is logged the first time it runs, and on every run
at level 5, so a process whose ``stderr`` is a pipe nobody reads (a notebook kernel, for example)
does not block once the pipe is full.

The primary purpose of the Logger is debugging, especially identifying the last called function
before a segmentation fault or crash (if the function includes a profiler tag).


Tracer
------

**Tracer** is a subtool designed to collect and display the execution time of each ``profiler_task``.

Example output (for ``cov_dense_batch``):

::

    Covariance::prepareSums 8.40us
    Covariance::prepareCrossProduct 1.26us
    computeSums.local 2.36us
    computeSums.reduce 630ns
    gemmSums 761ns
    Covariance::updateDenseCrossProductAndSums 316.42us
    CovarianceDenseBatchKernel::compute 782.13us

A top-level task, that is an algorithm call, is traced on every run. A nested task is traced the
first time its call path runs, and on every run at level 5. The analyzer has the totals.

The Tracer is particularly useful in Jupyter Notebooks or general profiling scenarios where performance insights are needed.


Analyzer
--------

**Analyzer** is a subtool that provides a hierarchical view of the full algorithm execution tree, along with time spent in each task.

Example output (for ``cov_dense_batch``):

::

    Algorithm tree analyzer
    |-- CovarianceDenseBatchKernel::compute time: 782.13us 100.00% 1 times in a sequential region
    |  |-- Covariance::prepareSums time: 19.51us 2.49% 1 times in a sequential region
    |  |-- Covariance::prepareCrossProduct time: 1.35us 0.17% 1 times in a sequential region
    |  |-- Covariance::updateDenseCrossProductAndSums time: 636.83us 81.42% 1 times in a sequential region
    |  |  |-- computeSums.local time: 2.54us 0.33% 32 times in a parallel region on 16 threads
    |  |  |-- computeSums.reduce time: 783ns 0.10% 16 times in a parallel region on 16 threads
    |  |  |-- gemmSums time: 767ns 0.10% 16 times in a parallel region on 16 threads
    |  |-- compute.finalizeCovariance time: 1.19us 0.15% 1 times in a sequential region
    |--(end)
    DAAL KERNEL_PROFILER: kernels total time 782.13us

Each row is a call path: the same task name under different parents gives different rows, and the
runs of a path are merged into one row with their count. A task that ran on a thread of a parallel
region is nested under the task that started the region, whichever thread ran it. Its row is marked
``parallel`` and shows the time of the busiest thread, so the times of the rows under a task do not
exceed the task's own time; ``on N threads`` is the number of threads that ran it.

The Analyzer helps visualize nested kernel calls and their contribution to the total runtime. It's especially helpful for understanding which tasks call which kernels and how much time each part consumes.

Reports on request
------------------

The report is printed when the process exits. A process that does not exit, such as a notebook
kernel, can ask for it at any point through a C entry point of ``libonedal_core``:

.. code-block:: python

    import ctypes
    core = ctypes.CDLL("libonedal_core.so")
    core._onedal_profiler_report(b"report.txt", 1)  # append the report to the file, then clear

The first argument is the file to append to, or ``None`` for ``stderr``; a non-zero second argument
drops everything recorded so far, so the next report covers only the calls made after it.



General Notes on Verbose Mode
-----------------------------

The mode is read from ``ONEDAL_VERBOSE`` once, at the first profiled call, and cannot be changed
afterwards in the same process. With the profiler off, each task costs one check of a cached flag.
