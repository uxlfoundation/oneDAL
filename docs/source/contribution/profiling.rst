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
The analyzer becomes available only at the end of the oneDAL workflow, when static objects are destroyed.

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


When enabled with timing, execution timing is taken synchronously, meaning previous kernel execution will block later kernels.

Logger
------

**Logger** is a subtool used to capture information about each launched kernel.

Example output (for ``cov_dense_batch``):

::

    auto oneapi::dal::covariance::backend::compute_kernel_dense_impl<float>::operator()(const descriptor_t &, const parameters_t &, const input_t &)::(anonymous class)::operator()() const [Float = float]
    Profiler task_name: allreduce_sums, q_

    auto oneapi::dal::covariance::backend::compute_kernel_dense_impl<float>::operator()(const descriptor_t &, const parameters_t &, const input_t &)::(anonymous class)::operator()() const [Float = float]
    Profiler task_name: gemm, q_

Each entry shows the ``__PRETTY_FUNCTION__`` along with the corresponding ``PROFILER_TASK`` and its arguments.

The primary purpose of the Logger is debugging — especially useful for identifying the last called function before a segmentation fault or crash (if the function includes a profiler tag).


Tracer
------

**Tracer** is a subtool designed to collect and display the execution time of each ``profiler_task``.

Example output (for ``cov_dense_batch``):

::

    Covariance::prepareSums 8.40us
    Covariance::prepareCrossProduct 1.26us
    THREADING computeSums.local finished on the main rank(time could be different for other ranks): 2.36us
    THREADING computeSums.reduce finished on the main rank(time could be different for other ranks): 630ns
    THREADING gemmSums finished on the main rank(time could be different for other ranks): 761ns
    THREADING Covariance::updateDenseCrossProductAndSums finished on the main rank(time could be different for other ranks): 316.42us

The Tracer is particularly useful in Jupyter Notebooks or general profiling scenarios where performance insights are needed.


Analyzer
--------

**Analyzer** is a subtool that provides a hierarchical view of the full algorithm execution tree, along with time spent in each task.

Example output (for ``cov_dense_batch``):

::

    Algorithm tree analyzer
    |-- CovarianceDenseBatchKernel::compute time: 782.13us 100.00% 1 times in a 0 region
    |   |-- Covariance::prepareSums time: 19.51us 2.49% 1 times in a 0 region
    |   |-- Covariance::prepareCrossProduct time: 1.35us 0.17% 1 times in a 0 region
    |   |-- Covariance::updateDenseCrossProductAndSums time: 636.83us 81.42% 1 times in a 1 region
    |   |-- computeSums.local time: 2.54us 0.33% 2 times in a 1 region
    |   |-- computeSums.reduce time: 783ns 0.10% 2 times in a 1 region
    |   |-- gemmSums time: 767ns 0.10% 1 times in a 1 region
    |   |-- compute.finalizeCovariance time: 1.19us 0.15% 1 times in a 0 region
    |---(end)

    ONEDAL KERNEL_PROFILER: kernels total time 501.60ms

The Analyzer helps visualize nested kernel calls and their contribution to the total runtime. It's especially helpful for understanding which tasks call which kernels and how much time each part consumes.



General Notes on Verbose Mode
-----------------------------

Important: This means that the analyzer will not work properly in interactive environments like Jupyter Notebooks, 
where the application may not fully terminate and static object destructors may never be called.
To use the analyzer, run your application as a standalone script.
