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

Working with User-defined Quality Metrics
=========================================

In addition to or instead of the metrics available in the library, you can use your own quality metrics. To do this:

#. Add your own implementation of the quality metrics algorithm and define Input and Result classes for that algorithm.

#. Register this new algorithm in the ``inputAlgorithms`` collection of the quality metric set.
   Also register the input objects for the new algorithm in the ``inputData`` collection of the quality metric set.

Use the unique key when registering the new algorithm and its input, and use the same key to obtain the computed results.
