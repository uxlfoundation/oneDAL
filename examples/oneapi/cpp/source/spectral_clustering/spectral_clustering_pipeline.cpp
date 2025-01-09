/*******************************************************************************
* Copyright contributors to the oneDAL project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/algo/kmeans_init.hpp"
#include "oneapi/dal/algo/spectral_embedding.hpp"
#include "oneapi/dal/io/csv.hpp"
#include <algorithm>
#include <math.h>

namespace dal = oneapi::dal;

int main(int argc, char const *argv[]) {
    double p = 0.01; // prunning parameter
    std::int64_t num_spks = 8; // dimension of spectral embeddings

    std::int64_t cluster_count = num_spks; // number of clusters
    std::int64_t max_iteration_count = 300; // max iterations number for K-Means
    std::int64_t n_init = 20; // number of K-means++ iterations
    double accuracy_threshold = 1e-4; // threshold for early stop in K-Means

    const auto voice_data_file_name =
        get_data_path("covcormoments_dense.csv"); // Dataset with original features

    std::cout << voice_data_file_name << std::endl;

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ voice_data_file_name });

    std::int64_t m = x_train.get_row_count();
    std::int64_t n_neighbors;

    if (m < 1000) {
        n_neighbors = std::min((std::int64_t)10, m - 2);
    }
    else {
        n_neighbors = (std::int64_t)(p * m);
    }

    const auto spectral_embedding_desc =
        dal::spectral_embedding::descriptor<>()
            .set_neighbor_count(n_neighbors)
            .set_component_count(num_spks)
            .set_result_options(dal::spectral_embedding::result_options::embedding |
                                dal::spectral_embedding::result_options::eigen_values);

    const auto spectral_embedding_result = dal::compute(spectral_embedding_desc, x_train);

    const auto spectral_embeddings =
        spectral_embedding_result.get_embedding(); // Matrix with spectral embeddings m * num_spks

    std::cout << "Spectral embeddings:\n" << spectral_embeddings << std::endl;

    std::cout << "Eigen values:\n" << spectral_embedding_result.get_eigen_values() << std::endl;

    const auto kmeans_init_desc =
        dal::kmeans_init::descriptor<float, dal::kmeans_init::method::plus_plus_dense>()
            .set_cluster_count(cluster_count)
            .set_local_trials_count(n_init);

    const auto kmeans_init_result = dal::compute(kmeans_init_desc, spectral_embeddings);

    const auto initial_centroids = kmeans_init_result.get_centroids();

    const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(cluster_count)
                                 .set_max_iteration_count(max_iteration_count)
                                 .set_accuracy_threshold(accuracy_threshold);

    const auto spectral_clustering_result =
        dal::train(kmeans_desc, spectral_embeddings, initial_centroids);

    std::cout << "Responses:\n" << spectral_clustering_result.get_responses() << std::endl;
    std::cout << "Centroids:\n"
              << spectral_clustering_result.get_model().get_centroids() << std::endl;
    return 0;
}
