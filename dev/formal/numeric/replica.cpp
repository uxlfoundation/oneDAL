/********************************************************************************
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
*********************************************************************************/

// Host replica of the arithmetic of basic_statistics GPU compute_by_blocks:
// block_processor_kernel (Welford per block), merge_blocks_kernel (first loop with
// stride L, then barrier tree), and the distributed branch of finalize.
// It is not the SYCL kernel: compiler, FMA contraction and device math may differ.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

template <typename F>
struct Part { F sum, sum2cent, mean; std::int64_t rc; };

template <typename F>
std::vector<Part<F>> process_blocks(const std::vector<F>& x, std::int64_t b0, std::int64_t b1,
                                    std::int64_t block_count) {
    const std::int64_t row_count = b1 - b0;
    const std::int64_t bs = (row_count + block_count - 1) / block_count;
    std::vector<Part<F>> out(block_count, Part<F>{ 0, 0, 0, 0 });
    for (std::int64_t blk = 0; blk < block_count; ++blk) {
        F sum = 0, mean = 0, s2c = 0;
        const std::int64_t f = bs * blk, l = std::min(row_count, f + bs);
        for (std::int64_t r = f; r < l; ++r) {
            F val = x[b0 + r];
            sum += val;
            const F delta = val - mean;
            const F rel_row = F(r - f);
            const F inv_n = F(1) / (rel_row + F(1));
            mean += delta * inv_n;
            s2c += delta * (val - mean);
        }
        out[blk] = { sum, s2c, 0, std::max<std::int64_t>(l - f, 0) };
    }
    return out;
}

// One merge step exactly as in merge_blocks_kernel; acc.mean is mrgmean, b.mean is block mean.
template <typename F>
void step(F& n, F& mean, F& s2c, F& sum, F bsum, F bs2c, F bmean, std::int64_t rcnt) {
    const F sum_n1n2 = n + F(rcnt);
    const F mul_n1n2 = n * F(rcnt);
    const F delta_scale = mul_n1n2 / sum_n1n2;
    const F mean_scale = F(1) / sum_n1n2;
    const F delta = bmean - mean;
    sum += bsum;
    s2c = s2c + bs2c + delta * delta * delta_scale;
    mean = (mean * n + bmean * F(rcnt)) * mean_scale;
    n = sum_n1n2;
}

struct Acc { double n, mean, s2c, sum; };

template <typename F>
Acc merge_kernel(const std::vector<Part<F>>& p, std::int64_t L, const std::string& order,
                 unsigned seed) {
    const std::int64_t B = p.size();
    if (order != "kernel") {  // alternative merge orders of the same pairwise formula
        std::vector<std::int64_t> idx(B);
        for (std::int64_t i = 0; i < B; ++i) idx[i] = i;
        if (order == "rev") std::reverse(idx.begin(), idx.end());
        if (order == "shuffle") { std::mt19937 g(seed); std::shuffle(idx.begin(), idx.end(), g); }
        F n = 0, mean = 0, s2c = 0, sum = 0;
        for (auto i : idx) step<F>(n, mean, s2c, sum, p[i].sum, p[i].sum2cent,
                                   p[i].sum / F(p[i].rc), p[i].rc);
        return { double(n), double(mean), double(s2c), double(sum) };
    }
    std::vector<F> ln(L, 0), lmean(L, 0), ls2c(L, 0), lsum(L, 0);
    std::vector<F> pn(L, 0), pmean(L, 0), ps2c(L, 0), psum(L, 0);  // private registers
    for (std::int64_t id = 0; id < L; ++id)
        for (std::int64_t i = id; i < B; i += L) {
            step<F>(pn[id], pmean[id], ps2c[id], psum[id], p[i].sum, p[i].sum2cent,
                    p[i].sum / F(p[i].rc), p[i].rc);
            ln[id] = pn[id]; lmean[id] = pmean[id]; ls2c[id] = ps2c[id]; lsum[id] = psum[id];
        }
    for (std::int64_t stride = std::min(L, B) / 2; stride > 0; stride /= 2)
        for (std::int64_t id = 0; id < stride; ++id) {
            const std::int64_t o = id + stride;
            step<F>(pn[id], pmean[id], ps2c[id], psum[id], lsum[o], ls2c[o], lmean[o],
                    std::int64_t(ln[o]));
            ln[id] = pn[id]; lmean[id] = pmean[id]; ls2c[id] = ps2c[id]; lsum[id] = psum[id];
        }
    return { double(pn[0]), double(pmean[0]), double(ps2c[0]), double(psum[0]) };
}

std::int64_t row_block_count(std::int64_t n) {  // get_row_block_count
    if (n < 5000) return 1;
    if (n < 10000) return 8;
    if (n < 20000) return 16;
    if (n < 50000) return 32;
    if (n < 100000) return 64;
    return 128;
}

template <typename F>
int run(const std::vector<double>& xd, std::int64_t forcedB, std::int64_t L,
        const std::string& order, unsigned seed, std::int64_t ranks) {
    std::vector<F> x(xd.begin(), xd.end());
    const std::int64_t N = x.size();
    std::vector<Acc> per;
    for (std::int64_t r = 0; r < ranks; ++r) {
        const std::int64_t b0 = N * r / ranks, b1 = N * (r + 1) / ranks;
        const std::int64_t B = forcedB > 0 ? forcedB : row_block_count(b1 - b0);
        if (B == 1) {  // compute_single_pass: one Welford block, same recurrence
            auto p = process_blocks<F>(x, b0, b1, 1);
            per.push_back({ double(p[0].rc), double(p[0].sum / F(p[0].rc)),
                            double(p[0].sum2cent), double(p[0].sum) });
        } else {
            per.push_back(merge_kernel<F>(process_blocks<F>(x, b0, b1, B), L, order, seed));
        }
    }
    double mean, var;
    if (ranks == 1) {
        mean = per[0].mean;
        var = double(F(per[0].s2c) / (F(per[0].n) - F(1)));
    } else {  // finalize, distributed branch: allreduce sum, then sum of per-rank terms
        F gsum = 0;
        for (auto& a : per) gsum += F(a.sum);
        const F inv_n = F(1.0 / double(N));
        const F mrgmean = gsum * inv_n;
        F gvar = 0;
        for (auto& a : per) {
            const F delta = mrgmean - F(a.mean);
            F s2c = F(a.s2c) + delta * delta * F(a.n);
            gvar += s2c / F(N - 1);
        }
        mean = mrgmean; var = gvar;
    }
    std::printf("%.17g %.17g\n", mean, var);
    return 0;
}

int main(int argc, char** argv) {
    // replica <file.f64> <float|double> <B or 0=auto> <L> <kernel|seq|rev|shuffle> <seed> <ranks>
    if (argc != 8) return 2;
    FILE* f = std::fopen(argv[1], "rb");
    std::vector<double> xd;
    double v;
    while (std::fread(&v, sizeof v, 1, f) == 1) xd.push_back(v);
    std::fclose(f);
    const std::string t = argv[2], order = argv[5];
    const auto B = std::atoll(argv[3]), L = std::atoll(argv[4]), R = std::atoll(argv[7]);
    const unsigned seed = std::atoi(argv[6]);
    return t == "float" ? run<float>(xd, B, L, order, seed, R) : run<double>(xd, B, L, order, seed, R);
}
