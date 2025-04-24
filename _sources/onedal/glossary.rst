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

.. _glossary:

=========
Glossary
=========

Machine learning terms
======================

.. glossary::
    :sorted:

    Categorical feature
        A :capterm:`feature` with a discrete domain. Can be :capterm:`nominal
        <nominal feature>` or :capterm:`ordinal <ordinal feature>`.

        **Synonyms:** discrete feature, qualitative feature

    Nominal feature
        A :capterm:`categorical feature` without ordering between values. Only
        equality operation is defined for nominal features.

        **Examples:** a person's gender, color of a car

    Classification
        A :capterm:`supervised machine learning problem <supervised learning>`
        of assigning :capterm:`labels <label>` to :capterm:`feature vectors
        <feature vector>`.

        **Examples:** predict what type of object is on the picture (a dog or a cat?),
        predict whether or not an email is spam

    Clustering
        An :capterm:`unsupervised machine learning problem <unsupervised learning>`
        of grouping :capterm:`feature vectors <feature vector>` into bunches, which
        are usually encoded as :capterm:`nominal <nominal feature>` values.

        **Example:** find big star clusters in the space images

    Continuous feature
        A :capterm:`feature` with values in a domain of real numbers. Can be
        :capterm:`interval <interval feature>` or :capterm:`ratio <ratio feature>`

        **Synonyms:** quantitative feature, numerical feature

        **Examples:** a person's height, the price of the house

    Dimensionality reduction
        A problem of transforming a set of :capterm:`feature vectors <feature
        vector>` from a high-dimensional space into a low-dimensional space
        while retaining meaningful properties of the original feature vectors.

    Feature
        A particular property or quality of a real object or an event. Has a
        defined type and domain. In machine learning problems, features are
        considered as input variable that are independent from each other.

        **Synonyms:** attribute, variable, input variable

    Feature vector
        A vector that encodes information about real object, an event or a group
        of objects or events. Contains at least one :capterm:`feature`.

        **Example:** A rectangle can be described by two features: its width and
        height

    Inference
        A process of applying a :capterm:`trained <Training>` :capterm:`model`
        to the :capterm:`dataset` in order to predict :capterm:`response` values
        based on input :capterm:`feature vectors <Feature vector>`.

        **Synonym:** prediction

    Inference set
        A :capterm:`dataset` used at the :capterm:`inference` stage.
        Usually without :capterm:`responses <Response>`.

    Interval feature
        A :capterm:`continuous feature` with values that can be compared, added or
        subtracted, but cannot be multiplied or divided.

        **Examples:** a time frame scale, a temperature in Celsius or Fahrenheit

    Label
        A :capterm:`response` with :capterm:`categorical <Categorical feature>` or
        :capterm:`ordinal <Ordinal feature>` values. This is an output in
        :capterm:`classification` and :capterm:`clustering` problems.

        **Example:** the spam-detection problem has a binary label indicating
        whether the email is spam or not

    Model
        An entity that stores information necessary to run :capterm:`inference`
        on a new :capterm:`dataset`. Typically a result of a :capterm:`training`
        process.

        **Example:** in linear regression algorithm, the model contains weight
        values for each input feature and a single bias value

    Nu-classification
        An SVM-specific :capterm:`classification` problem where :math:`\nu` parameter is used
        instead of :math:`C`. :math:`\nu` is an upper bound on the fraction
        of training errors and a lower bound of the fraction of the support vector.

    Nu-regression
        An SVM-specific :capterm:`regression` problem where :math:`\nu` parameter is used
        instead of :math:`\epsilon`. :math:`\nu` is an upper bound on the fraction
        of training errors and a lower bound of the fraction of the support vector.

    Observation
        A :capterm:`feature vector` and zero or more :capterm:`responses<Response>`.

        **Synonyms:** instance, sample

    Result options:
        Result options are entities that mimic C++ enums. They are used to specify which results
        of an algorithm should be computed. The use of result options may alter the
        default algorithm flow and result in performance differences.
        In general, fewer results to compute means faster performance.
        An error is thrown when you use an invalid set of result options or try to access the results
        that are not yet computed.

        **Example:** k-NN Classification algorithm can perform classification
        and also return indices and distances to the nearest observations as a
        result option.

    Ordinal feature
        A :capterm:`categorical feature` with defined operations of equality and
        ordering between values.

        **Example:** student's grade

    Outlier
        :capterm:`Observation` which is significantly different from the other
        observations.

    Ratio feature
        A :capterm:`continuous feature` with defined operations of equality,
        comparison, addition, subtraction, multiplication, and division.
        Zero value element means the absence of any value.

        **Example:** the height of a tower

    Regression
        A :capterm:`supervised machine learning problem <Supervised learning>` of
        assigning :capterm:`continuous <Continuous feature>`
        :capterm:`responses<Response>` for :capterm:`feature vectors <Feature vector>`.

        **Example:** predict temperature based on weather conditions

    Response
        A property of some real object or event which dependency from
        :capterm:`feature vector` need to be defined in :capterm:`supervised learning`
        problem. While a :capterm:`feature` is an input in the machine learning
        problem, the response is one of the outputs can be made by the
        :capterm:`model` on the :capterm:`inference` stage.

        **Synonym:** dependent variable

    Search
        A kNN-specific optimization problem of finding the point in a given set
        that is the closest to the given points.

    Supervised learning
        :capterm:`Training` process that uses a :capterm:`dataset` with information
        about dependencies between :capterm:`features <Feature>` and
        :capterm:`responses <Response>`. The goal is to get a :capterm:`model` of
        dependencies between input :capterm:`feature vector` and
        :capterm:`responses <Response>`.

    Training
        A process of creating a :capterm:`model` based on information extracted
        from a :capterm:`training set`. Resulting :capterm:`model` is selected in
        accordance with some quality criteria.

    Partial Training
        A process of computing partial results based on information extracted
        from a :capterm:`training set`.

    Finalize Training
        A process of creating a :capterm:`model` based on information extracted
        from partial results. Resulting :capterm:`model` is selected
        according to some quality criteria.

    Training set
        A :capterm:`dataset` used at the :capterm:`training` stage to create a
        :capterm:`model`.

    Unsupervised learning
        :capterm:`Training` process that uses a :capterm:`training set` with no
        :capterm:`responses <Response>`. The goal is to find hidden patters inside
        :capterm:`feature vectors <Feature vector>` and dependencies between them.

    CSV file
        A comma-separated values file (csv) is a type of a text file. Each line in a CSV file is a record containing fields that are separated by the delimiter.
        Fields can be of a numerical or a text format. Text usually refers to categorical values.
        By default, the delimiter is a comma, but, generally, it can be any character.
        For more details, `see <https://en.wikipedia.org/wiki/Comma-separated_values>`_.

Graph analytics terms
======================

.. glossary::
    :sorted:

    Adjacency
        A vertex :math:`u` is adjacent to vertex :math:`v` if they are joined by
        an edge.

    Adjacency matrix
        An :math:`n \times n` matrix :math:`A_G` for a graph :math:`G` whose vertices
        are explicitly ordered :math:`(v_1, v_2, ..., v_n)`,

        .. math::
            \mathrm{A_G}=\begin{cases}
               1, \text{where } v_i \text{ and } v_j \text{ adjacent} \\
               0, \text{otherwise.}
            \end{cases}

    Attribute
        A value assigned to :capterm:`graph`, vertex or edge.
        Can be numerical (:capterm:`weight`), string or any other custom data type.

    Component
        A :capterm:`connected<Connected graph>` :capterm:`subgraph<Subgraph>` :math:`H` of graph :math:`G` such that no subgraph
        of :math:`G` that properly contains :math:`H` is connected [Gross2014]_.

    Connected graph
        A :capterm:`graph` is connected if there is a :capterm:`walk` between every pair of its vertices [Gross2014]_.

    Edge index
        The index :math:`i` of an edge :math:`e_i` in an edge set :math:`E=\{e_1, e_2,  ..., e_m\}`
        of :capterm:`graph` :math:`G`. Can be an integer value.

    Directed graph
        A :capterm:`graph` where each edge is an ordered pair :math:`(u, v)`
        of vertices. :math:`v` is designated as the tail, and :math:`u` is designated
        as the head.

    Graph
        An object :math:`G=(V;E)` that consists of two sets, :math:`V` and :math:`E`,
        where :math:`V` is a finite nonempty set, :math:`E` is a finite set that may
        be empty, and the elements of :math:`E` are two-element subsets of :math:`V`.
        :math:`V` is called a set of vertices, :math:`E` is called a set of edges [Gross2014]_.

    Subgraph
        A graph :math:`H = (V'; E')` is called a subgraph of graph :math:`G = (V; E)` if
        :math:`V' \subseteq V; E' \subseteq E` and :math:`V'` contain all endpoints of all the
        edges in :math:`E'` [Gross2014]_.

    Induced subgraph on the edge set
        Each subset :math:`E' \subseteq E` defines a unique :capterm:`subgraph <Subgraph>` :math:`H' = (V'; E')` of graph
        :math:`G = (V; E)`, where :math:`V'` consists of only those vertices that are the endpoints of the
        edges in :math:`E'`. The subgraph :math:`H` is called an induced subgraph of :math:`G` on the
        edge set :math:`E'` [Gross2014]_.

    Induced subgraph on the vertex set
        Each subset :math:`V' \subseteq V` defines a unique :capterm:`subgraph <Subgraph>`
        :math:`H = (V'; E')` of graph :math:`G = (V; E)`, where :math:`E'` consists of those edges
        whose endpoints are in :math:`V'`. The subgraph :math:`H` is called an induced subgraph of :math:`G` on the vertex
        set :math:`V'` [Gross2014]_.

    Self-loop
        An edge that joins a vertex to itself.

    Topology
        A :capterm:`graph` without :capterm:`attributes <Attribute>`.

    Undirected graph
        A :capterm:`graph` where each edge is an unordered pair :math:`(u, v)` of vertices.

    Unweighted graph
        A :capterm:`graph` where all vertices and all edges has no :capterm:`weights <Weight>`.

    Vertex index
        The index :math:`i` of a vertex :math:`v_i` in a vertex set :math:`V=\{v_1, v_2,  ..., v_n\}`
        of :capterm:`graph` :math:`G`. Can be an integer value.

    Walk
        An alternating sequence of vertices and edges such that for each edge,
        one endpoint precedes and the other succeeds that edge in the sequence [Gross2014]_.

    Weight
        A numerical value assigned to vertex, edge or graph.

    Weighted graph
        A :capterm:`graph` where all vertices or all
        edges have :capterm:`weights <Weight>.`

|short_name| terms
======================

.. glossary::
    :sorted:

    Accessor
        A |short_name| concept for an object that provides access to the
        data of another object in the special :capterm:`data format`. It abstracts
        data access from interface of an object and provides uniform access to
        the data stored in objects of different types.

    Batch mode
        The computation mode for an algorithm in |short_name|, where all the
        data needed for computation is available at the start and fits the
        memory of the device on which the computations are performed.

    Builder
        A |short_name| concept for an object that encapsulates the creation
        process of another object and enables its iterative creation.

    Contiguous data
        Data that are stored as one contiguous memory block. One of the
        characteristics of a :capterm:`data format`.

    CSR data
        A Compressed Sparse Row (CSR) data is the sparse matrix representation.
        Data with values of a single :capterm:`data type` and the same set of
        available operations defined on them. One of the characteristics of a
        :capterm:`data format`.
        This representation stores the non-zero elements of a matrix in three
        arrays.
        The arrays describe the sparse matrix :math:`A` as follows:

        - The array values contain non-zero elements of the matrix row-by-row.
        - The element number ``j`` of the ``columns_indices`` array encodes
          the column index in the matrix :math:`A` for the jth element
          of the array values.
        - The element number ``i`` of the ``row_offsets`` array encodes
          the index in the array values corresponding to the first non-zero
          element in rows indexed ``i`` or greater.
          The last element in the array ``row_offsets`` encodes the number
          of non-zero elements in the matrix :math:`A`.

        |short_name| supports zero-based and one-based indexing.

    Data format
        Representation of the internal structure of the data.

        **Examples:** data can be stored in array-of-structures or
        compressed-sparse-row format

    Data layout
        A characteristic of :capterm:`data format` which describes the
        order of elements in a :capterm:`contiguous data` block.

        **Example:** row-major format, where elements are stored row by row

    Data type
        An attribute of data used by a compiler to store and access them.
        Includes size in bytes, encoding principles, and available operations
        (in terms of a programming language).

        **Examples:** ``int32_t``, ``float``, ``double``

    Dataset
        A collection of data in a specific data format.

        **Examples:** a collection of observations, a :capterm:`graph`

    Flat data
        A block of :capterm:`contiguous <contiguous data>` :capterm:`homogeneous
        <homogeneous data>` data.

    Getter
        A method that returns the value of the private member variable.

        **Example**:

        .. code-block:: cpp

            std::int64_t get_row_count() const;

    Heterogeneous data
        Data which contain values either of different :capterm:`data types <Data
        type>` or different sets of operations defined on them. One of the
        characteristics of a :capterm:`data format`.

        **Example:** A :capterm:`dataset` with 100
        :capterm:`observations <Observation>` of three :capterm:`interval features <Interval
        feature>`. The first two features are of float32 :capterm:`data type`, while the
        third one is of float64 data type.

    Homogeneous data
        Data with values of single :capterm:`data type` and the same set of
        available operations defined on them. One of the characteristics of a
        :capterm:`data format`.

        **Example:** A :capterm:`dataset` with 100
        :capterm:`observations <Observation>` of three  :capterm:`interval features <Interval
        feature>`, each of type float32


    Immutability
        The object is immutable if it is not possible to change its state after
        creation.

    Metadata
        Information about logical and physical structure of an object. All
        possible combinations of metadata values present the full set of
        possible objects of a given type. Metadata do not expose information
        that is not a part of a type definition, e.g. implementation details.

        **Example:** :capterm:`table` object can contain three :capterm:`nominal features
        <Nominal feature>` with 100 :capterm:`observations <Observation>` (logical
        part of metadata). This object can store data as sparse csr array and
        provides direct access to them (physical part)

    Online mode
        The computation mode for an algorithm in |short_name|, where the
        data needed for computation becomes available in parts over time.

    Reference-counted object
        A copy-constructible and copy-assignable |short_name| object which
        stores the number of references to the unique implementation. Both copy
        operations defined for this object are lightweight, which means that
        each time a new object is created, only the number of references is
        increased. An implementation is automatically freed when the number of
        references becomes equal to zero.

    Setter
        A method that accepts the only parameter and assigns its value to the
        private member variable.

        **Example**:

        .. code-block:: cpp

            void set_row_count(std::int64_t row_count);


    Table
        A |short_name| concept for a :capterm:`dataset` that contains only
        numerical data, :capterm:`categorical <Categorical feature>` or
        :capterm:`continuous <Continuous feature>`. Serves as a transfer of data
        between user's application and computations inside |short_name|.
        Hides details of :capterm:`data format` and generalizes access to the data.

    Workload
        A problem of applying a |short_name| algorithm to a :capterm:`dataset`.

Common oneAPI terms
===================

.. glossary::
    :sorted:

    API
        Application Programming Interface

    DPC++
        Data Parallel C++ (DPC++) is a high-level language designed for data
        parallel programming productivity. DPC++ is based on :term:`SYCL*
        <SYCL>` from the Khronos* Group to support data parallelism and
        heterogeneous programming.

    Host/Device
        OpenCL [OpenCLSpec]_ refers to CPU that controls the connected GPU
        executing kernels.

    JIT
        Just in Time Compilation --- compilation during execution of a program.

    Kernel
        Code written in OpenCL [OpenCLSpec]_ or :term:`SYCL` and executed on a
        GPU device.

    SPIR-V
        Standard Portable Intermediate Representation - V is a language for
        intermediate representation of compute kernels.

    SYCL
        SYCL(TM) [SYCLSpec]_ --- high-level programming model for OpenCL(TM)
        that enables code for heterogeneous processors to be written in a
        "single-source" style using completely standard C++.

Distributed computational mode terms
====================================

.. glossary::
    :sorted:

    SPMD
        Single Program, Multiple Data (SPMD) is a technique employed to achieve parallelism.
        In SPMD model, multiple autonomous processors simultaneously execute the same program at independent points.

    Communicator
        A |short_name| concept for an object that is used to perform inter-process collective
        operations

    Communicator backend
        A particular library providing collective operations.

        **Examples:** oneCCL, oneMPI

