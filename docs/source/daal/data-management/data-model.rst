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

Data Model
==========

The Data Model component of the |full_name| (|short_name|)
provides classes for model representation. The
model mimics the actual data and represents it in a compact way so
that you can use the library when the actual data is missing,
incomplete, noisy or unavailable.

There are two categories of models in the library: Regression models
and Classification models. Regression models are used to predict the
values of dependent variables (responses) by observing independent
variables. Classification models are used to predict to which
sub-population (class) a given observation belongs.

A set of parameters characterizes each model. |short_name| model
classes provide interfaces to access these parameters. It also
provides the corresponding classes to train models, that is, to
estimate model parameters using training data sets. As soon as a
model is trained, it can be used for prediction and cross-validation.
For this purpose, the library provides the corresponding prediction
classes.
