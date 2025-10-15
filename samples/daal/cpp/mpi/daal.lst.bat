@echo off
rem ============================================================================
rem Copyright 2017 Intel Corporation
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.
rem ============================================================================

::  Content:
::     oneAPI Data Analytics Library samples list
::******************************************************************************

set MPI_SAMPLE_LIST=svd_fast_distributed_mpi                      ^
                    qr_fast_distributed_mpi                       ^
                    linear_regression_norm_eq_distributed_mpi     ^
                    linear_regression_qr_distributed_mpi          ^
                    pca_correlation_dense_distributed_mpi         ^
                    pca_correlation_csr_distributed_mpi           ^
                    pca_svd_distributed_mpi                       ^
                    covariance_dense_distributed_mpi              ^
                    covariance_csr_distributed_mpi                ^
                    multinomial_naive_bayes_dense_distributed_mpi ^
                    multinomial_naive_bayes_csr_distributed_mpi   ^
                    kmeans_dense_distributed_mpi                  ^
                    kmeans_csr_distributed_mpi                    ^
                    kmeans_init_dense_distributed_mpi             ^
                    kmeans_init_csr_distributed_mpi               ^
                    implicit_als_csr_distributed_mpi              ^
                    ridge_regression_norm_eq_distributed_mpi      ^
                    dbscan_dense_distributed_mpi
