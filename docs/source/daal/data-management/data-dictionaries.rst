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

Data Dictionaries
=================

A data dictionary is the metadata that describes features of a data
set. The NumericTableFeature and DataSourceFeature structures
describe a particular feature within a dictionary of the associated
numeric table and data source respectively. These structures specify:

-  Whether the feature is continuous, categorical, or ordinal
-  Underlying data types (double, integer, and so on) used to
   represent feature values

The DataSourceFeature structure also specifies:

-  Possible values for a categorical feature
-  The feature name

The DataSourceDictionary class is a data dictionary that describes
raw data associated with the corresponding data source. The
NumericTableDictionary class is a data dictionary that describes
in-memory numeric data associated with the corresponding numeric
table. Both classes provide generic methods for dictionary
manipulation, such as accessing a particular data feature, setting
and retrieving the number of features, and adding a new feature.
Respective DataSource and NumericTable classes have generic
dictionary manipulation methods, such as getDictionary() and
setDictionary().

To create a dictionary from the data source context, you can do one
of the following:

-  Set the doDictionaryFromContext flag in the DataSource
   constructor.
-  Call to the createDictionaryFromContext() method.

Examples
********

C++:

-  :cpp_example:`datasource/datastructures_aos.cpp`
-  :cpp_example:`datasource/datastructures_soa.cpp`
-  :cpp_example:`datasource/datastructures_homogen.cpp`

