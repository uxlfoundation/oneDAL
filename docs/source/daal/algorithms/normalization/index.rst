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

Normalization
=============

Normalization is a set of algorithms intended to transform data before feeding it to some classes
of algorithms, for example, classifiers [James2013]_.
Normalization may improve computation accuracy and efficiency.
Different rules can be used to normalize data.
In |short_name|, two techniques to normalize data are implemented: z-score and min-max.

.. toctree::
   :maxdepth: 1

   z-score.rst
   min-max.rst
