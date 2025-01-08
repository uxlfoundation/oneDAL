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

Subgraph Isomorphism algorithm receives a target graph :math:`G` and a pattern graph :math:`H` as input
and searches the target graph for subgraphs that are isomorphic to the pattern graph. The algorithm returns
the mappings of the pattern graph vertices onto the target graph vertices.

.. |si_compute|     replace::   :ref:`Computing              <subgraph_isomorphism_compute>`
.. |si_fast|        replace::   :ref:`fast                   <subgraph_isomorphism_fast>`
.. |si_api|         replace::   :ref:`graph_matching(...)    <subgraph_isomorphism_t_api>`
.. |si_api_input|   replace::   :ref:`graph_matching_input   <subgraph_isomorphism_t_api_input>`
.. |si_api_result|  replace::   :ref:`graph_matching_result  <subgraph_isomorphism_t_api_result>`

================ =========================== ============ ================= =================
 **Operation**     **Computational methods**           **Programming Interface**
---------------- --------------------------- ------------------------------------------------
  |si_compute|             |si_fast|            |si_api|    |si_api_input|    |si_api_result|
================ =========================== ============ ================= =================
