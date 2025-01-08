.. Copyright 2020 Intel Corporation
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

Quality Metrics
===============

In |short_name|, a quality metric is a numerical characteristic or a set of connected numerical characteristics
that represents the qualitative aspect of the result returned by an algorithm:
a computed statistical estimate, model, or result of decision making.

A common set of quality metrics can be defined for some training and prediction algorithms.

A typical workflow with quality metric set is the following:

#.  Create a quality metric set object to compute quality metrics.

    - Set specific parameters for the algorithms.
    - Use the ``useDefaultMetrics`` flag to specify whether the default or user-defined quality metrics should be computed.

#. Get an input collection object using ``QualityMetricsId`` of a specific algorithm.
#. Set data to the input collection using the algorithm's ``InputId``.
#. Perform computation.
#. Get the resulting collection of quality metrics using the algorithm's ``ResultId``.

.. note::
    For values of ``InputId``, ``Parameters``, ``QualityMetricsId``, ``ResultId``,
    refer to the description of a specific algorithm.

Quality metrics are optional. They are computed when the computation is explicitly requested.

.. toctree::
   :maxdepth: 2

   default-metric-set.rst
   user-defined.rst
