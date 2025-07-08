#===============================================================================
# Copyright 2014 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

##  Content:
##     oneDAL configuration file for the Sphinx documentation builder
##******************************************************************************

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'oneDAL'
copyright = 'contributors to the oneDAL project' # pylint: disable=redefined-builtin

# The full version, including alpha/beta/rc tags
# release = '2021'

rst_prolog = """
.. include:: /substitutions_common.txt
.. include:: /substitutions_specific.txt
"""

# for substitutions in code blocks and sphinx-prompts:
substitutions = [
    ('|short_name|', 'oneDAL'),
    ('|daal_in_code|', 'daal')
    ]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# sys.path.insert(0, path_relative_to_repo_root('source/elements/oneDAL'))

extensions = ['sphinx_prompt',
              'sphinx_substitution_extensions',
              'sphinx.ext.extlinks',
              'sphinx_tabs.tabs',
              'dalapi',
              'sphinx.ext.githubpages',
              'notfound.extension']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["opt-notice.rst", 'daal/data-management/numeric-tables/*.rst', 'get-started/*.rst',
                    'daal/algorithms/dbscan/distributed-steps/*',
                    'daal/algorithms/kmeans/includes/*',
                    'daal/includes/*', 'onedal/algorithms/.*/includes/*', 'index-toc.rst']

extlinks = {
    'cpp_example': ('https://github.com/uxlfoundation/oneDAL/tree/main/examples/daal/cpp/source/%s', None),
    'daal4py_example': ('https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/%s', None),
    'daal4py_sklearnex_example': ('https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/sklearnex/%s', None),
    'cpp_sample': ('https://github.com/uxlfoundation/oneDAL/tree/main/samples/daal/cpp/%s', None)
}

# -- Options for HTML output -------------------------------------------------


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme = 'sphinx_book_theme'
html_favicon = '_static/favicons.png'

# Theme options
html_theme_options = {
    'repository_url': 'https://github.com/uxlfoundation/oneDAL',
    'path_to_docs': 'docs/source',
    'use_issues_button': True,
    'use_edit_page_button': True,
    'repository_branch': 'master',
    "logo": {
        "text": "oneDAL Documentation",
        "image_light": "_static/uxl-foundation-logo-horizontal-color.png",
        "image_dark": "_static/uxl-foundation-logo-horizontal-white.png",
    },
}


# oneDAL project directory is needed for `dalapi` extension
onedal_enable_listing = False
onedal_relative_doxyfile_dir = '../doxygen/oneapi'
onedal_relative_sources_dir = '../../cpp/oneapi/dal'

# not found 404 page

notfound_urls_prefix = '/oneDAL/'

# ignore these missing references during a doc build
nitpick_ignore = [
    # top level namespace
    ('cpp:identifier', 'dal'),
    ('cpp:identifier', 'base'),
    # method
    ('cpp:identifier', 'method'),
    ('cpp:identifier', 'method::by_default'),
    ('cpp:identifier', 'Method'),
    # task
    ('cpp:identifier', 'task'),
    ('cpp:identifier', 'task::by_default'),
    ('cpp:identifier', 'Task'),
    # detail
    ('cpp:identifier', 'dal::detail'),
    ('cpp:identifier', 'dal::detail::empty_delete<T>'),
    ('cpp:identifier', 'detail'),
    ('cpp:identifier', 'detail::descriptor_base<>'),
    ('cpp:identifier', 'detail::descriptor_base<>::float_t'),
    ('cpp:identifier', 'detail::descriptor_base<>::method_t'),
    ('cpp:identifier', 'detail::descriptor_base<>::task_t'),
    ('cpp:identifier', 'detail::table_builder'),
    ('cpp:identifier', 'detail::is_table_impl_v<ImplType>'),
    ('cpp:identifier', 'detail::is_homogen_table_impl_v<ImplType>'),
    ('cpp:identifier', 'detail::enable_if_classification_t<T>'),
    ('cpp:identifier', 'detail::enable_if_regression_t<T>'),
    ('cpp:identifier', 'detail::descriptor_base<>::kernel_t'),
    # data types
    ('cpp:identifier', 'int64_t'),
    ('cpp:identifier', 'data_t'),
    ('cpp:identifier', 'kernel_t'),
    ('cpp:identifier', 'distance_t'),
    # knn
    ('cpp:identifier', 'knn'),
    ('cpp:identifier', 'knn::descriptor'),
    ('cpp:identifier', 'knn::train_result'),
    ('cpp:identifier', 'knn::train_input'),
    ('cpp:identifier', 'knn::infer_result'),
    ('cpp:identifier', 'knn::infer_input'),
    ('cpp:identifier', 'task::search'),
    ('cpp:identifier', 'detail::enable_if_brute_force_t<M>'),
    ('cpp:identifier', 'detail::enable_if_search_t<T>'),
    ('cpp:identifier', 'detail::enable_if_not_classification_t<T>'),
    # minkowski_distance
    ('cpp:identifier', 'minkowski_distance'),
    ('cpp:identifier', 'minkowski_distance::descriptor'),
    ('cpp:identifier', 'oneapi::dal::minkowski_distance'),
    ('cpp:identifier', 'oneapi::dal::minkowski_distance::descriptor<Float>'),
    # chebyshev_distance
    ('cpp:identifier', 'chebyshev_distance'),
    ('cpp:identifier', 'chebyshev_distance::descriptor'),
    # correlation_distance
    ('cpp:identifier', 'correlation_distance'),
    ('cpp:identifier', 'correlation_distance::descriptor'),
    ('cpp:identifier', 'correlation_distance::compute_result'),
    ('cpp:identifier', 'correlation_distance::compute_input'),
    # kmeans
    ('cpp:identifier', 'kmeans'),
    ('cpp:identifier', 'kmeans::descriptor'),
    ('cpp:identifier', 'kmeans::train_result'),
    ('cpp:identifier', 'kmeans::train_input'),
    ('cpp:identifier', 'kmeans::infer_result'),
    ('cpp:identifier', 'kmeans::infer_input'),
    ('cpp:identifier', 'i'),
    ('cpp:identifier', 'kmeans::model'),
    ('cpp:identifier', 'infer_input'),
    ('cpp:identifier', 'infer_input::model'),
    ('cpp:identifier', 'infer_input::model::centroids'),
    ('cpp:identifier', ),
    # kmeans_init
    ('cpp:identifier', 'kmeans_init'),
    ('cpp:identifier', 'local_trials'),
    ('cpp:identifier', 'kmeans_init::descriptor'),
    ('cpp:identifier', 'kmeans_init::compute_input'),
    ('cpp:identifier', 'kmeans_init::compute_result'),
    ('cpp:identifier', 'compute'),
    # objective_function
    ('cpp:identifier', 'objective_function'),
    ('cpp:identifier', 'objective_t'),
    ('cpp:identifier', 'objective_function::descriptor'),
    ('cpp:identifier', 'objective_function::compute_result'),
    ('cpp:identifier', 'objective_function::compute_input'),
    # logloss_objective
    ('cpp:identifier', 'logloss_objective'),
    ('cpp:identifier', 'logloss_objective::descriptor'),
    # logistic_regression
    ('cpp:identifier', 'logistic_regression'),
    ('cpp:identifier', 'oneapi::dal::logistic_regression'),
    ('cpp:identifier', 'logistic_regression::descriptor'),
    ('cpp:identifier', 'logistic_regression::train_result'),
    ('cpp:identifier', 'logistic_regression::train_input'),
    ('cpp:identifier', 'logistic_regression::infer_result'),
    ('cpp:identifier', 'logistic_regression::infer_input'),
    ('cpp:identifier', 'optimizer_t'),
    # linear_regression
    ('cpp:identifier', 'linear_regression'),
    ('cpp:identifier', 'oneapi::dal::linear_regression'),
    ('cpp:identifier', 'linear_regression::descriptor'),
    ('cpp:identifier', 'linear_regression::train_result'),
    ('cpp:identifier', 'linear_regression::train_input'),
    ('cpp:identifier', 'linear_regression::infer_result'),
    ('cpp:identifier', 'linear_regression::infer_input'),
    # newton_cg
    ('cpp:identifier', 'newton_cg'),
    ('cpp:identifier', 'oneapi::dal::newton_cg'),
    ('cpp:identifier', 'oneapi::dal::newton_cg::descriptor<Float>'),
    ('cpp:identifier', 'tol'),
    ('cpp:identifier', 'maxiter'),
    # pca
    ('cpp:identifier', 'pca'),
    ('cpp:identifier', 'pca::descriptor'),
    ('cpp:identifier', 'pca::train_result'),
    ('cpp:identifier', 'pca::train_input'),
    ('cpp:identifier', 'pca::infer_result'),
    ('cpp:identifier', 'pca::infer_input'),
    # svm
    ('cpp:identifier', 'svm'),
    ('cpp:identifier', 'svm::descriptor'),
    ('cpp:identifier', 'svm::train_result'),
    ('cpp:identifier', 'svm::train_input'),
    ('cpp:identifier', 'svm::infer_result'),
    ('cpp:identifier', 'svm::infer_input'),
    ('cpp:identifier', 'Kernel'),
    ('cpp:identifier', 'oneapi::dal::svm'),
    ('cpp:identifier', 'oneapi::dal::svm::method'),
    ('cpp:identifier', 'oneapi::dal::svm::method::v1'),
    ('cpp:identifier', 'Kernel'),
    ('cpp:identifier', 'task::regression'),
    ('cpp:identifier', 'task::classification'),
    ('cpp:identifier', 'detail::enable_if_regression_t<T>'),
    # linear kernel
    ('cpp:identifier', 'linear_kernel'),
    ('cpp:identifier', 'linear_kernel::descriptor'),
    ('cpp:identifier', 'linear_kernel::compute_result'),
    ('cpp:identifier', 'linear_kernel::compute_input'),
    # polynomial kernel
    ('cpp:identifier', 'polynomial_kernel'),
    ('cpp:identifier', 'polynomial_kernel::descriptor'),
    ('cpp:identifier', 'polynomial_kernel::compute_result'),
    ('cpp:identifier', 'polynomial_kernel::compute_input'),
    # rbf kernel
    ('cpp:identifier', 'rbf_kernel'),
    ('cpp:identifier', 'rbf_kernel::descriptor'),
    ('cpp:identifier', 'rbf_kernel::compute_result'),
    ('cpp:identifier', 'rbf_kernel::compute_input'),
    # sigmoid kernel
    ('cpp:identifier', 'sigmoid_kernel'),
    ('cpp:identifier', 'sigmoid_kernel::descriptor'),
    ('cpp:identifier', 'sigmoid_kernel::compute_result'),
    ('cpp:identifier', 'sigmoid_kernel::compute_input'),
    # decision forest
    ('cpp:identifier', 'decision_forest'),
    ('cpp:identifier', 'decision_forest::infer_result'),
    ('cpp:identifier', 'decision_forest::infer_input'),
    ('cpp:identifier', 'decision_forest::train_result'),
    ('cpp:identifier', 'decision_forest::train_input'),
    ('cpp:identifier', 'decision_forest::descriptor'),
    ('cpp:identifier', 'variable_importance_mode'),
    ('cpp:identifier', 'variable_importance_mode::none'),
    ('cpp:identifier', 'variable_importance_mode::mda_raw'),
    ('cpp:identifier', 'variable_importance_mode::mda_scaled'),
    ('cpp:identifier', 'error_metric_mode'),
    ('cpp:identifier', 'error_metric_mode::none'),
    ('cpp:identifier', 'error_metric_mode::out_of_bag_error'),
    ('cpp:identifier', 'error_metric_mode::out_of_bag_error_per_observation'),
    # covariance
    ('cpp:identifier', 'covariance'),
    ('cpp:identifier', 'covariance::descriptor'),
    ('cpp:identifier', 'covariance::compute_result'),
    ('cpp:identifier', 'covariance::compute_input'),
    ('cpp:identifier', 'oneapi::dal::covariance'),
    ('cpp:identifier', 'oneapi::dal::covariance::task'),
    ('cpp:identifier', 'oneapi::dal::covariance::task::v1'),
    ('cpp:identifier', 'oneapi::dal::covariance::task::v1::compute'),
    ('cpp:identifier', 'oneapi::dal::covariance::method'),
    ('cpp:identifier', 'oneapi::dal::covariance::method::v1'),
    ('cpp:identifier', 'oneapi::dal::covariance::method::v1::dense'),
    # basic statistics
    ('cpp:identifier', 'basic_statistics'),
    ('cpp:identifier', 'basic_statistics::descriptor'),
    ('cpp:identifier', 'basic_statistics::compute_result'),
    ('cpp:identifier', 'basic_statistics::compute_input'),
    ('cpp:identifier', 'oneapi::dal::basic_statistics'),
    ('cpp:identifier', 'oneapi::dal::basic_statistics::task'),
    ('cpp:identifier', 'oneapi::dal::basic_statistics::task::v1'),
    ('cpp:identifier', 'oneapi::dal::basic_statistics::task::v1::compute'),
    ('cpp:identifier', 'oneapi::dal::basic_statistics::method'),
    ('cpp:identifier', 'oneapi::dal::basic_statistics::method::v1'),
    ('cpp:identifier', 'oneapi::dal::basic_statistics::method::v1::dense'),
    # dbscan
    ('cpp:identifier', 'dbscan'),
    ('cpp:identifier', 'dbscan::descriptor'),
    ('cpp:identifier', 'dbscan::compute_result'),
    ('cpp:identifier', 'dbscan::compute_input'),
    ('cpp:identifier', 'oneapi::dal::dbscan'),
    ('cpp:identifier', 'oneapi::dal::dbscan::task'),
    ('cpp:identifier', 'oneapi::dal::dbscan::task::v1'),
    ('cpp:identifier', 'oneapi::dal::dbscan::task::v1::compute'),
    ('cpp:identifier', 'oneapi::dal::dbscan::method'),
    ('cpp:identifier', 'oneapi::dal::dbscan::method::v1'),
    ('cpp:identifier', 'oneapi::dal::dbscan::method::v1::brute_force'),
    # common for algorithms
    ('cpp:identifier', 'result'),
    # common for result options
    ('cpp:identifier', 'result_option_id_base'),
    ('cpp:identifier', 'result_option_id'),
    # tables
    ('cpp:identifier', 'table'),
    ('cpp:identifier', 'row_count'),
    ('cpp:identifier', 'column_count'),
    ('cpp:identifier', 'is_readonly'),
    ('cpp:identifier', 'range'),
    ('cpp:identifier', 'empty_table_kind'),
    ('cpp:identifier', 'data_layout'),
    ('cpp:identifier', 'data_layout::row_major'),
    ('cpp:identifier', 'data_layout::unknown'),
    ('cpp:identifier', 'feature_type'),
    ('cpp:identifier', 'data_type'),
    ('cpp:identifier', 'table_metadata'),
    ('cpp:identifier', 'mutable_data'),
    ('cpp:identifier', 'data'),
    ('cpp:identifier', 'count'),
    ('cpp:identifier', 'sparse_indexing'),
    ('cpp:identifier', 'sparse_indexing::zero_based'),
    ('cpp:identifier', 'sparse_indexing::one_based'),
    # array
    ('cpp:identifier', 'T'),
    ('cpp:identifier', 'I'),
    ('cpp:identifier', 'array'),
    ('cpp:identifier', 'array_d'),
    ('cpp:identifier', 'array_i'),
    ('cpp:identifier', 'impl_t'),
    ('cpp:identifier', 'array<T>'),
    ('cpp:identifier', 'array<Y>'),
    ('cpp:identifier', 'has_mutable_data'),
    # csv
    ('cpp:identifier', 'csv'),
    ('cpp:identifier', 'read'),
    ('cpp:identifier', 'read_options'),
    ('cpp:identifier', 'default_read_options'),
    ('cpp:identifier', 'default_delimiter'),
    ('cpp:identifier', 'Object'),
    # oneapi
    ('cpp:identifier', 'oneapi'),
    ('cpp:identifier', 'oneapi::dal'),
    # oneapi - kmeans
    ('cpp:identifier', 'oneapi::dal::kmeans'),
    ('cpp:identifier', 'oneapi::dal::kmeans::task'),
    # oneapi - decision_forest
    ('cpp:identifier', 'oneapi::dal::decision_forest'),
    ('cpp:identifier', 'oneapi::dal::decision_forest::task'),
    ('cpp:identifier', 'oneapi::dal::decision_forest::task::v1'),
    # oneapi - svm
    ('cpp:identifier', 'oneapi::dal::svm'),
    ('cpp:identifier', 'oneapi::dal::svm::method'),
    ('cpp:identifier', 'oneapi::dal::svm::task'),

    # graphs
    ('cpp:identifier', 'vertex_size_type<Graph>'),
    ('cpp:identifier', 'edge_size_type<Graph>'),
    ('cpp:identifier', 'vertex_type<Graph>'),
    ('cpp:identifier', 'vertex_edge_size_type<Graph>'),
    ('cpp:identifier', 'const_vertex_edge_range_type<Graph>'),
    ('cpp:identifier', 'vertex_outward_edge_size_type<Graph>'),
    ('cpp:identifier', 'const_vertex_outward_edge_range_type<Graph>'),
    ('cpp:identifier', 'edge_user_value_type<Graph>'),
    ('cpp:identifier', 'directed_adjacency_vector_graph'),
    ('cpp:identifier', 'undirected_adjacency_vector_graph'),
    ('cpp:identifier', 'Allocator'),
    ('cpp:identifier', 'Graph'),
    ('cpp:identifier', 'subgraph_isomorphism'),
    ('cpp:identifier', 'kind::induced'),
    ('cpp:identifier', 'kind::non_induced'),
    ('cpp:identifier', 'preview'),
    ('cpp:identifier', 'connected_components'),

    # sycl
    ('cpp:identifier', 'event_vector'),
    ('cpp:identifier', 'sycl'),
    ('cpp:identifier', 'sycl::event'),
    ('cpp:identifier', 'sycl::queue'),
    ('cpp:identifier', 'sycl::range'),
    ('cpp:identifier', 'sycl::usm'),
    ('cpp:identifier', 'sycl::usm::alloc'),
    ('cpp:identifier', 'sycl::usm::alloc::shared'),

    # backend primitives - data management
    ('cpp:identifier', 'array_t'),
    ('cpp:identifier', 'axis_count'),
    ('cpp:identifier', 'ndarray'),
    ('cpp:identifier', 'ndindex<axis_count>'),
    ('cpp:identifier', 'ndorder'),
    ('cpp:identifier', 'ndorder::c'),
    ('cpp:identifier', 'ndorder::f'),
    ('cpp:identifier', 'ndshape'),
    ('cpp:identifier', 'ndshape<new_axis_count>'),
    ('cpp:identifier', 'ndview'),
    ('cpp:identifier', 'order'),
    ('cpp:identifier', 'sycl::range<1>'),
    ('cpp:identifier', 'sycl::range<2>'),
    ('cpp:identifier', 'shape_t'),
    ('cpp:identifier', 'shared_t')
]
