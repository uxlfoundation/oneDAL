// Comprehensive benchmark for ALL threader_for function variants
// Tests: threader_for, threader_for_simple, threader_for_blocked,
//        static_threader_for, threader_for_optional, threader_for_break
// With multiple N sizes and work patterns
//
// Compile: icpx -std=c++17 -O2 -I. -Icpp/daal -Icpp/daal/src -L__release_lnx/daal/latest/lib/intel64 -lonedal_core -lonedal_thread -lpthread -Wl,-rpath,__release_lnx/daal/latest/lib/intel64 grain_bench.cpp -o grain_bench
// Run: ./grain_bench

#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <cstring>
#include <functional>

#include "cpp/daal/src/threading/threading.h"

double median_of(std::vector<double>& t) {
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

template <typename Fn>
double measure(Fn fn, int reps = 7) {
    fn(); // warmup
    std::vector<double> times;
    for (int r = 0; r < reps; ++r) {
        auto s = std::chrono::high_resolution_clock::now();
        fn();
        auto e = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(e - s).count());
    }
    return median_of(times);
}

void header(const char* name) {
    printf("\n========== %s ==========\n", name);
    printf("  %-8s", "grain");
}

void header_Ns(const int64_t* Ns, int count) {
    for (int i = 0; i < count; i++)
        printf("  N=%-12lld", (long long)Ns[i]);
    printf("\n");
}

void row(int64_t grain, const double* ms, int count, const double* baselines) {
    printf("  %-8lld", (long long)grain);
    for (int i = 0; i < count; i++) {
        if (baselines)
            printf("  %7.2f (%4.1fx) ", ms[i], ms[i] / baselines[i]);
        else
            printf("  %7.2f ms     ", ms[i]);
    }
    printf("\n");
}

int main() {
    size_t nth = daal::threader_env()->getNumberOfThreads();
    printf("=== Comprehensive Threader Function Benchmark ===\n");
    printf("Threads: %zu\n", nth);

    const int64_t grains[] = {1, 4, 16, 64, 256, 1024, 4096, 16384};
    const int NG = 8;

    // =====================================================================
    // 1. threader_for — store-only (array init)
    // =====================================================================
    {
        int64_t Ns[] = {1000, 10000, 100000, 1000000, 10000000};
        int NN = 5;
        header("threader_for — store only");
        header_Ns(Ns, NN);

        double baselines[5];
        for (int g = 0; g < NG; g++) {
            double ms[5];
            for (int n = 0; n < NN; n++) {
                std::vector<int64_t> d(Ns[n]);
                ms[n] = measure([&]() {
                    daal::threader_for(Ns[n], grains[g], [&](int64_t i) { d[i] = 0; });
                });
                if (g == 0) baselines[n] = ms[n];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 2. threader_for — light math (a[i] = b[i] + c[i])
    // =====================================================================
    {
        int64_t Ns[] = {1000, 10000, 100000, 1000000, 10000000};
        int NN = 5;
        header("threader_for — light math (add)");
        header_Ns(Ns, NN);

        double baselines[5];
        for (int g = 0; g < NG; g++) {
            double ms[5];
            for (int n = 0; n < NN; n++) {
                std::vector<double> a(Ns[n], 1.0), b(Ns[n], 2.0), c(Ns[n]);
                ms[n] = measure([&]() {
                    daal::threader_for(Ns[n], grains[g], [&](int64_t i) {
                        c[i] = a[i] + b[i];
                    });
                });
                if (g == 0) baselines[n] = ms[n];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 3. threader_for — medium math (inner loop F elements per iter)
    // =====================================================================
    {
        int64_t Ns[] = {1000, 10000, 100000};
        int NN = 3;
        int64_t F = 50;
        header("threader_for — inner loop F=50 (vSub pattern)");
        header_Ns(Ns, NN);

        double baselines[3];
        for (int g = 0; g < NG; g++) {
            double ms[3];
            for (int n = 0; n < NN; n++) {
                std::vector<double> data(Ns[n] * F, 3.14);
                std::vector<double> result(Ns[n] * F);
                std::vector<double> mean(F, 1.0);
                ms[n] = measure([&]() {
                    daal::threader_for(Ns[n], grains[g], [&](int64_t i) {
                        for (int64_t j = 0; j < F; j++)
                            result[i * F + j] = data[i * F + j] - mean[j];
                    });
                });
                if (g == 0) baselines[n] = ms[n];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 4. threader_for — heavy math (sin+cos per element)
    // =====================================================================
    {
        int64_t Ns[] = {1000, 10000, 100000, 1000000};
        int NN = 4;
        header("threader_for — heavy math (sin+cos)");
        header_Ns(Ns, NN);

        double baselines[4];
        for (int g = 0; g < NG; g++) {
            double ms[4];
            for (int n = 0; n < NN; n++) {
                std::vector<double> d(Ns[n], 1.0);
                ms[n] = measure([&]() {
                    daal::threader_for(Ns[n], grains[g], [&](int64_t i) {
                        d[i] = std::sin(static_cast<double>(i)) + std::cos(static_cast<double>(i));
                    });
                });
                if (g == 0) baselines[n] = ms[n];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 5. threader_for_simple — store only
    // =====================================================================
    {
        int64_t Ns[] = {1000, 10000, 100000, 1000000, 10000000};
        int NN = 5;
        header("threader_for_simple — store only");
        header_Ns(Ns, NN);

        double baselines[5];
        for (int g = 0; g < NG; g++) {
            double ms[5];
            for (int n = 0; n < NN; n++) {
                std::vector<int64_t> d(Ns[n]);
                ms[n] = measure([&]() {
                    daal::threader_for_simple(Ns[n], grains[g], [&](int64_t i) { d[i] = 0; });
                });
                if (g == 0) baselines[n] = ms[n];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 6. threader_for_simple — light math
    // =====================================================================
    {
        int64_t Ns[] = {1000, 10000, 100000, 1000000, 10000000};
        int NN = 5;
        header("threader_for_simple — light math (add)");
        header_Ns(Ns, NN);

        double baselines[5];
        for (int g = 0; g < NG; g++) {
            double ms[5];
            for (int n = 0; n < NN; n++) {
                std::vector<double> a(Ns[n], 1.0), b(Ns[n], 2.0), c(Ns[n]);
                ms[n] = measure([&]() {
                    daal::threader_for_simple(Ns[n], grains[g], [&](int64_t i) {
                        c[i] = a[i] + b[i];
                    });
                });
                if (g == 0) baselines[n] = ms[n];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 7. threader_for_blocked — store (block-level init)
    // =====================================================================
    {
        int64_t Ns[] = {1000, 10000, 100000, 1000000, 10000000};
        int NN = 5;
        header("threader_for_blocked — block store");
        header_Ns(Ns, NN);

        double baselines[5];
        for (int g = 0; g < NG; g++) {
            double ms[5];
            for (int n = 0; n < NN; n++) {
                std::vector<int64_t> d(Ns[n]);
                ms[n] = measure([&]() {
                    daal::threader_for_blocked(Ns[n], grains[g], [&](int64_t begin, int64_t count) {
                        for (int64_t i = begin; i < begin + count; i++) d[i] = 0;
                    });
                });
                if (g == 0) baselines[n] = ms[n];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 8. threader_for_blocked — heavy math
    // =====================================================================
    {
        int64_t Ns[] = {1000, 10000, 100000, 1000000};
        int NN = 4;
        header("threader_for_blocked — heavy math (sin+cos)");
        header_Ns(Ns, NN);

        double baselines[4];
        for (int g = 0; g < NG; g++) {
            double ms[4];
            for (int n = 0; n < NN; n++) {
                std::vector<double> d(Ns[n], 1.0);
                ms[n] = measure([&]() {
                    daal::threader_for_blocked(Ns[n], grains[g], [&](int64_t begin, int64_t count) {
                        for (int64_t i = begin; i < begin + count; i++)
                            d[i] = std::sin(static_cast<double>(i)) + std::cos(static_cast<double>(i));
                    });
                });
                if (g == 0) baselines[n] = ms[n];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 9. static_threader_for — store only
    // =====================================================================
    {
        int64_t Ns[] = {1000, 10000, 100000, 1000000, 10000000};
        int NN = 5;
        header("static_threader_for — store only (no grain param)");
        header_Ns(Ns, NN);

        double ms[5];
        for (int n = 0; n < NN; n++) {
            std::vector<int64_t> d(Ns[n]);
            ms[n] = measure([&]() {
                daal::static_threader_for(Ns[n], [&](int64_t i, size_t) { d[i] = 0; });
            });
        }
        printf("  static  ");
        for (int n = 0; n < NN; n++)
            printf("  %7.2f ms     ", ms[n]);
        printf("\n");
    }

    // =====================================================================
    // 10. threader_for_optional — store only
    // =====================================================================
    {
        int64_t Ns[] = {1000, 10000, 100000, 1000000, 10000000};
        int NN = 5;
        header("threader_for_optional — store only");
        header_Ns(Ns, NN);

        double baselines[5];
        for (int g = 0; g < NG; g++) {
            double ms[5];
            for (int n = 0; n < NN; n++) {
                std::vector<int64_t> d(Ns[n]);
                ms[n] = measure([&]() {
                    daal::threader_for_optional(Ns[n], grains[g], [&](int64_t i) { d[i] = 0; });
                });
                if (g == 0) baselines[n] = ms[n];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 11. threader_for_break — store with early exit
    // =====================================================================
    {
        int64_t Ns[] = {1000, 10000, 100000, 1000000, 10000000};
        int NN = 5;
        header("threader_for_break — store (no actual break)");
        header_Ns(Ns, NN);

        double baselines[5];
        for (int g = 0; g < NG; g++) {
            double ms[5];
            for (int n = 0; n < NN; n++) {
                std::vector<int64_t> d(Ns[n]);
                ms[n] = measure([&]() {
                    daal::threader_for_break(Ns[n], grains[g], [&](int64_t i, bool&) { d[i] = 0; });
                });
                if (g == 0) baselines[n] = ms[n];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 12. threader_for — atomic accumulate (graph pattern)
    // =====================================================================
    {
        int64_t Ns[] = {10000, 100000, 1000000, 10000000};
        int NN = 4;
        int64_t V = 100000;
        header("threader_for — atomic increment (degree count)");
        header_Ns(Ns, NN);

        double baselines[4];
        for (int g = 0; g < NG; g++) {
            double ms[4];
            for (int n = 0; n < NN; n++) {
                std::vector<int32_t> src(Ns[n]), dst(Ns[n]);
                for (int64_t i = 0; i < Ns[n]; i++) { src[i] = i % V; dst[i] = (i*7+13) % V; }
                std::vector<std::atomic<int32_t>> deg(V);
                ms[n] = measure([&]() {
                    for (auto& d : deg) d.store(0, std::memory_order_relaxed);
                    daal::threader_for(Ns[n], grains[g], [&](int64_t i) {
                        deg[src[i]].fetch_add(1, std::memory_order_relaxed);
                        deg[dst[i]].fetch_add(1, std::memory_order_relaxed);
                    });
                });
                if (g == 0) baselines[n] = ms[n];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 13. threader_for — cross-product merge (triangular loop)
    // =====================================================================
    {
        int64_t Ns[] = {100, 500, 1000, 2000};
        int NN = 4;
        header("threader_for — triangular inner loop (cross product)");
        header_Ns(Ns, NN);

        double baselines[4];
        for (int g = 0; g < NG; g++) {
            double ms[4];
            for (int n = 0; n < NN; n++) {
                int64_t F = Ns[n];
                std::vector<double> cross(F * F, 0.0), partial(F * F, 0.001);
                ms[n] = measure([&]() {
                    daal::threader_for(F, grains[g], [&](int64_t i) {
                        for (int64_t j = 0; j <= i; j++)
                            cross[i * F + j] += partial[i * F + j];
                    });
                });
                if (g == 0) baselines[n] = ms[n];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 14. threader_for — row-wise normalize (zscore pattern)
    // =====================================================================
    {
        int64_t Ns[] = {1000, 10000, 100000};
        int64_t Fs[] = {10, 50, 100, 500};
        int NN = 4;
        header("threader_for — zscore normalize, N=100K, varying F");
        printf("  %-8s", "grain");
        for (int f = 0; f < NN; f++)
            printf("  F=%-12lld", (long long)Fs[f]);
        printf("\n");

        int64_t N = 100000;
        double baselines[4];
        for (int g = 0; g < NG; g++) {
            double ms[4];
            for (int f = 0; f < NN; f++) {
                int64_t F = Fs[f];
                std::vector<double> data(N * F, 3.14), result(N * F);
                std::vector<double> mean(F, 1.0), invS(F, 0.5);
                ms[f] = measure([&]() {
                    daal::threader_for(N, grains[g], [&](int64_t i) {
                        for (int64_t j = 0; j < F; j++)
                            result[i * F + j] = (data[i * F + j] - mean[j]) * invS[j];
                    });
                });
                if (g == 0) baselines[f] = ms[f];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 15. threader_for — block-heavy work (nBlocks small, heavy per block)
    //     Simulates: decision forest predict, SVM, blocked GEMM patterns
    // =====================================================================
    {
        int64_t Ns[] = {8, 20, 50, 100, 200};
        int NN = 5;
        header("threader_for — heavy per-block (nBlocks small)");
        header_Ns(Ns, NN);

        double baselines[5];
        for (int g = 0; g < NG; g++) {
            double ms[5];
            for (int n = 0; n < NN; n++) {
                int64_t blockSize = 50000;
                std::vector<std::vector<double>> blocks(Ns[n], std::vector<double>(blockSize, 1.0));
                ms[n] = measure([&]() {
                    daal::threader_for(Ns[n], grains[g], [&](int64_t b) {
                        double sum = 0;
                        for (int64_t j = 0; j < blockSize; j++)
                            sum += std::sin(blocks[b][j]);
                        blocks[b][0] = sum;
                    });
                });
                if (g == 0) baselines[n] = ms[n];
            }
            row(grains[g], ms, NN, baselines);
        }
    }

    // =====================================================================
    // 16. threader_for_blocked vs threader_for comparison
    //     Same work, different chunking strategies
    // =====================================================================
    {
        int64_t N = 10000000;
        printf("\n========== threader_for vs threader_for_blocked vs threader_for_simple (N=10M, add) ==========\n");
        printf("  %-8s  %-18s  %-18s  %-18s\n", "grain", "threader_for", "for_blocked", "for_simple");

        for (int g = 0; g < NG; g++) {
            std::vector<double> a(N, 1.0), b(N, 2.0), c(N);

            double t1 = measure([&]() {
                daal::threader_for(N, grains[g], [&](int64_t i) { c[i] = a[i] + b[i]; });
            });
            double t2 = measure([&]() {
                daal::threader_for_blocked(N, grains[g], [&](int64_t begin, int64_t count) {
                    for (int64_t i = begin; i < begin + count; i++) c[i] = a[i] + b[i];
                });
            });
            double t3 = measure([&]() {
                daal::threader_for_simple(N, grains[g], [&](int64_t i) { c[i] = a[i] + b[i]; });
            });

            printf("  %-8lld  %7.2f ms         %7.2f ms         %7.2f ms\n",
                   (long long)grains[g], t1, t2, t3);
        }
    }

    printf("\n=== Done ===\n");
    return 0;
}
