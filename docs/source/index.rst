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

.. daal_rst documentation master file, created by
   sphinx-quickstart on Mon May 20 15:10:10 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |spec| replace:: |short_name| specification
.. _spec: https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.3-rev-1/elements/onedal/source/#


|full_name| (|short_name|)
------------------------------------------------------

|full_name| (|short_name|) is a C++ library (powering the `Extension for Scikit-learn in Python <https://github.com/uxlfoundation/scikit-learn-intelex>`__)
which implements accelerated machine learning routines for tabular data (e.g. linear regression, K-means clustering, random forests, etc.) for CPUs, GPUs,
and multi-node distributed setups.

It provides highly optimized algorithmic building blocks for all stages of data analytics
(preprocessing, transformation, analysis, modeling, validation, and
decision making) in batch, online, and distributed processing modes of
computation. The library provides two different sets of C++ interfaces: :ref:`oneAPI and DAAL <oneapi_vs_daal>`.

For general information, refer to |onedal-landing|_.

.. _oneapi_vs_daal:

oneAPI vs. DAAL Interfaces
==========================

- :ref:`oneapi_dal_guide` are based on open |spec|_ and are currently under an active development.
  They work on various hardware but only a limited set of algorithms is available at the moment.

- :ref:`daal_guide` are CPU-only interfaces that provide implementations for a wide range of algorithms.

.. include:: index-toc.rst
