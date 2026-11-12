# Formal verification experiments: basic_statistics

Two small experiments that measure what TLA+ and Lean 4 cost and what they find
on real oneDAL code. They do not verify the library. Both cover the same code:
the GPU `basic_statistics` dense compute path,
`cpp/oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel_dense_impl_dpc.cpp`.
The numeric part also runs the CPU (DAAL) `low_order_moments` merge.

| Directory | Tool | What it checks |
| --- | --- | --- |
| `tla/` | TLA+ / TLC | Execution protocol of `compute_by_blocks` -> `merge_blocks` |
| `lean/` | Lean 4 + Mathlib | Exact-arithmetic semantics of the count/mean/M2 merge |
| `numeric/` | C++ replica, daal4py | Floating-point behaviour of the same merges |

## A. TLA+: `compute_by_blocks` -> `merge_blocks`

### Code modelled

| Model element | Code |
| --- | --- |
| `SubmitK1`, `K1Write` | `compute_by_blocks`: `q_.submit` of `block_processor_kernel`, writes `ndbuf` |
| `SubmitK2`, `K2Ready` | `merge_blocks`: `q_.submit` with `cgh.depends_on(deps)`, `deps = { last_event }` |
| `Load` | `merge_blocks_kernel` first loop `for (i = id; i < block_count; i += local_size)` |
| `Barrier` | `sycl::group_barrier(item.get_group())` at the top of each tree round |
| `Reduce` | `if (stride > id)` merge `l*_ptr[id + stride]`; `if (0 < id)` store own slot |
| `Finish` | `if (0 == id)` write `r*_ptr[group_id]` |
| `Wait` | `last_event.wait_and_throw()` at the end of `merge_blocks` |
| `Free` | `ndbuf`, a local of `compute_by_blocks` passed to `merge_blocks` by `&&`, destroyed when `compute_by_blocks` returns: `usm_deleter` calls `sycl::free(ptr, queue)` (`cpp/oneapi/dal/backend/memory.hpp`), which does not wait for kernels |

One work-group of K2 is modelled (one column; groups share no memory). `L` is
`device_max_sg_size`, `B` is `block_count`.

### Runtime semantics used (SYCL 2020)

- `handler::depends_on(e)`: the command group does not start before `e` completes.
- `event::wait_and_throw()`: returns after the command completes.
- `sycl::free(ptr, q)` on USM: the spec does not say it waits, and freeing memory a
  running command still uses is undefined. The model takes the conservative reading:
  free does not wait.
- `group_barrier(g)`: every work-item of `g` reaches it before any continues, and
  it orders work-group local memory accesses.
- Not modelled, because neither the runtime contract nor the code has them:
  crashes, retries, partial kernel completion, device loss.

### Properties

`NoUseAfterFree`, `NoReadBeforeWrite` (K2 reads only blocks K1 wrote),
`NoStaleLocalRead` (a tree round reads a partner slot only after the partner
finished the previous round), `ResultCoversAllBlocks` (item 0 merged every
block exactly once), `ResultReadAfterK2`, `Termination` (liveness, and TLC
deadlock check: barriers never block forever).

### Results (`TLA2TOOLS=path/to/tla2tools.jar tla/run.sh NAME L B MUT`)

| Run | L | B | Mutation | Result | Distinct states | Time |
| --- | --- | --- | --- | --- | --- | --- |
| base_L2_B2 | 2 | 2 | none | pass | 34 | <1 s |
| base_L4_B1 | 4 | 1 | none | pass | 114 | <1 s |
| base_L4_B4 | 4 | 4 | none | pass | 450 | <1 s |
| base_L4_B8 | 4 | 8 | none | pass | 1,105 | <1 s |
| base_L8_B8 | 8 | 8 | none | pass | 85,730 | 5 s |
| base_L4_B16 | 4 | 16 | none | pass | 132,705 | 5 s |
| pre_L4_B6 | 4 | 6 | none (B >= L, not 2^k) | pass | 609 | <1 s |
| mut_nodep | 4 | 4 | drop `depends_on(deps)` | `NoReadBeforeWrite` violated | | <1 s |
| mut_nowait | 4 | 4 | drop `wait_and_throw()` | `NoUseAfterFree` violated, 10-state trace | | <1 s |
| mut_nobarrier | 4 | 4 | drop `group_barrier` | `NoStaleLocalRead` violated | | <1 s |
| pre_L4_B3 | 4 | 3 | none (B < L, not 2^k) | `ResultCoversAllBlocks` violated: block 2 never merged | | <1 s |

Traces are in `tla/out/mut_nowait.trace.txt` and `tla/out/pre_L4_B3.trace.txt`.
In `mut_nowait` the host frees `ndbuf` (state 5) while K2 is still
`submitted`, and work-item 0 then loads block 0 (state 10).

`pre_L4_B3` is a latent precondition, not a reachable bug: the tree starts at
`stride = min(L, B) / 2`, so when `B < L` and `B` is not a power of two the last
`B - 2^floor(log2 B)` slots are never read. Today `get_row_block_count` returns
only 1, 8, 16, 32, 64 or 128, and sub-group sizes on current Intel GPUs are 8, 16
or 32 (assumed here, not checked by the code), so the
precondition `B >= L or B = 2^k` holds. An assertion in `merge_blocks` would
make it explicit. A Python check over every `row_count` in `[1, 2e6)` also
found no input where `block_processor_kernel` creates fewer non-empty blocks
than `block_count` (empty blocks would give `0/0` in the kernel, see B).

Not checked against the implementation: no GPU is available on the box this
ran on, so no SYCL run confirms these properties, and the mutations were made
in the model, not in the kernel. The distributed `finalize` (allreduce phases)
is not modelled; its collective sequence depends only on the compile-time
`List`, so every rank issues the same sequence.

## B. Lean 4: count/mean/M2 merge

`lean/OnedalStats/Moments.lean`, Lean `v4.15.0`, Mathlib `v4.15.0`.

| Definition / theorem | Statement | Code |
| --- | --- | --- |
| `merge` | `merge_blocks_kernel` step: `N = n1 + n2`, `delta = mean2 - mean1`, `m2 = m2a + m2b + delta*delta*(n1*n2/N)`, `mean = (mean1*n1 + mean2*n2)*(1/N)` | `compute_kernel_dense_impl_dpc.cpp` merge loop and tree |
| `merge_ofList` | `merge (ofList xs) (ofList ys) = ofList (xs ++ ys)`: the merge gives the count, mean and M2 of the concatenated sample | same |
| `merge_empty_left/right` | `empty = (0, 0, 0)` is neutral for every `Valid` partial result | contract only, see below |
| `merge_valid` | merge keeps `Valid` (n >= 0; n = 0 forces mean = M2 = 0) | |
| `merge_assoc` | merge is associative on `Valid` partial results | with `merge_comm`: any merge order or tree shape gives the same exact result |
| `merge_comm` | `merge a b = merge b a`, no hypotheses | |
| `kwayM2_correct` | `sum_r (M2_r + n_r * (mean - mean_r)^2) = M2` of all data | distributed branch of `finalize` |
| `daal_term_eq` | `c*(m1*m1 + m2*m2 - 2*m1*m2) = c*((m2 - m1)*(m2 - m1))` | DAAL CPU merge term, see C |
| `mergeBad_wrong` | dropping `/N` from `delta_scale` is wrong: blocks `[0]` and `[2, 4]` give M2 = 20, not 8 | negative variant |

Check: `lake build` (17 s after the Mathlib cache), then
`lake env lean AxiomCheck.lean`. Every theorem depends only on `propext`,
`Classical.choice` and `Quot.sound` (Lean's standard axioms, which Mathlib's
real numbers need). There is no `sorry`, `admit`, `axiom` or `native_decide`.

What the proofs mean: these are statements about real numbers. They are not a
proof of the C++ or SYCL code, and floating-point merge is not associative.
In Lean `x / 0 = 0`, so `merge empty s = s` holds in the model; in IEEE
arithmetic the kernel computes `mean = sum / rcnt = 0/0 = NaN` for an empty
block, so empty blocks are outside the kernel's contract (the replica runs
with forced empty blocks return NaN). The Lean source is ASCII only (`Real`,
`Ne`, `next =>`), per the repo rule.

## C. Numeric experiment

`numeric/experiment.py` (results in `results.json`, `summary.md`) compares:

- `replica` / `replica_fma`: `numeric/replica.cpp`, a host copy of the kernel
  arithmetic (`block_processor_kernel` Welford, `merge_blocks_kernel`, distributed
  `finalize`), built with `g++ -O2 -ffp-contract=off` and with
  `-ffp-contract=fast -march=native`. Partitions: `B` auto (from
  `get_row_block_count`), 8, 128; `L` 16, 32; merge order kernel tree, reversed,
  shuffled; 1 or 4 ranks. This is not the SYCL kernel: icpx defaults, device
  `sqrt` and FMA use can differ.
- Build: `g++ -O2 -ffp-contract=off -o replica replica.cpp` and
  `g++ -O2 -ffp-contract=fast -march=native -o replica_fma replica.cpp`; run
  `python experiment.py <dir with the binaries>` with `daal==2025.10.0` and `mpmath==1.3.0` installed.
- `daal4py`: real oneDAL CPU `low_order_moments` (pip `daal==2025.10.0`), batch
  and streaming with 1, 3, 64, 1000 chunks.
- Reference: exact rationals (`fractions`) over the exact binary inputs of each run.

Worst relative error of the variance over all supported configurations (double):

| Data | replica (GPU arithmetic) | daal4py batch / 1 chunk | daal4py >= 3 chunks |
| --- | --- | --- | --- |
| N(0,1), n=20000 | 5e-16 | 2e-15 | 5e-16 |
| 1e4 + N(0,1e-2), n=20000 | 3e-11 | 2e-11 | 1.8e-6 |
| 1e8 + N(0,1e-3), n=20000 | 2.7e-6 | 1.9e-6 | 1.9e5 (some runs give a negative variance, e.g. -0.085 vs 9.7e-7) |
| 1e8 + N(0,1e-3), n=150000 | 2.8e-6 | 4e-7 | 1.1e5 |

Single precision cannot hold `1e8 + 1e-3` (ulp is 8), so float rows for that
data measure nothing useful; they are in `summary.md`. Numeric tests do not
establish an error bound.

### Defect found: DAAL CPU online and distributed merge

`low_order_moments_impl.i` (online merge and distributed step 2) and
`low_order_moments_estimates_online.i` add
`coeff * (mean1*mean1 + mean2*mean2 - 2*mean1*mean2)`. Over the reals this is
`coeff * (mean2 - mean1)^2` (`daal_term_eq`), so the Lean theorems cannot tell
the two apart. In double, with means near 1e8 the squares are near 1e16 and
the difference cancels to rounding noise of size `coeff * ulp(1e16)`.
`numeric/cancellation.py` emulates both spellings: at offset 1e8 the expanded
form has relative error up to 3.7e5, the factored form up to 3e-7.

The fix is not in this branch: branch `fix/lom-merge-cancellation` (separate PR)
replaces the term with
`(mean2 - mean1) * (mean2 - mean1)` at the five sites and adds
`basic_statistics online merge with a large offset` to
`cpp/oneapi/dal/algo/basic_statistics/test/online.cpp`. With
`bazel test --config=host`, that test fails before the fix (M2 off by 492,
true value about 1e-3) and passes after; all five
`//cpp/oneapi/dal/algo/basic_statistics:tests` targets pass. The daal4py
numbers above are from the released 2025.10.0 wheel, not from the patched build.

## Summary

| Claim | Status |
| --- | --- |
| Pairwise and k-way merge formulas give the statistics of the union, merge is associative and commutative, `(0,0,0)` is neutral | proved in Lean over the reals |
| Dependency, lifetime and barrier protocol of `merge_blocks` | checked by TLC for L <= 8, B <= 16; not checked on a GPU |
| Removing `depends_on`, `wait_and_throw` or `group_barrier` breaks it | TLC counterexamples, model mutations only |
| Tree reduction drops blocks when `B < L` and `B` is not a power of two | TLC counterexample; unreachable with today's `get_row_block_count` |
| A host replica of the GPU kernel arithmetic stays within ~3e-6 relative variance error at offset 1e8 in double | host replica only, not the device |
| DAAL CPU online merge loses the variance at large offsets | reproduced with daal4py 2025.10.0; fix and red/green test in the separate PR |
| Distributed DAAL step 2 merge (same formula) | same edit in the separate PR; no SPMD run checked it |

## Effort and next candidates

Roughly, for one engineer with an agent: TLA+ model 2-3 h, Lean proofs 3 h
(most of it on `field_simp` case splits for zero counts), numeric harness 2 h,
the DAAL fix and test 1 h. TLC runs take seconds; `lake build` takes 17 s once
the Mathlib cache (about 5 GB) is downloaded. The useful find came from the
numeric differential run, guided by the exact-arithmetic spec; the proofs by
themselves cannot see floating-point cancellation.

Maintenance: the TLA+ mapping table has to be updated by hand when
`merge_blocks` changes. The Lean file depends only on `merge`'s formula and on
Mathlib; Mathlib upgrades break tactics roughly once per release.

Next candidates: the same merge in `covariance` and `low_order_moments` GPU
(`cpp/oneapi/dal/backend/primitives/stat`), the SPMD `finalize` allreduce
sequence with ranks that hold zero rows, and USM lifetime in other kernels that
free temporaries right after `submit`.
