.. Copyright 2021 Intel Corporation
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

Training
--------

::

   pca::model<> run_training(const table& data) {
      const auto pca_desc = pca::descriptor<float>{}
         .set_component_count(5)
         .set_deterministic(true);

      const auto result = train(pca_desc, data);

      print_table("means", result.get_means());
      print_table("variances", result.get_variances());
      print_table("eigenvalues", result.get_eigenvalues());
      print_table("eigenvectors", result.get_eigenvectors());

      return result.get_model();
   }

Inference
---------

::

   table run_inference(const pca::model<>& model,
                     const table& new_data) {
      const auto pca_desc = pca::descriptor<float>{}
         .set_component_count(model.get_component_count());

      const auto result = infer(pca_desc, model, new_data);

      print_table("labels", result.get_transformed_data());
   }