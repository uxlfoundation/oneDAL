# Decision Forest GPU training — end-to-end overview

This document describes how the histogram-based decision forest **training** runs
on GPU in oneDAL: the entrypoint, the call graph, the per-level algorithm, and
where the key per-node quantities (row counts, impurity, class histograms, and
sample-weight sums) are produced and propagated.

All paths are under
`cpp/oneapi/dal/algo/decision_forest/backend/gpu/`.

---

## 1. Entrypoint and dispatch

The public `train(...)` call is routed through the algorithm's `train_ops`
dispatcher to the GPU kernel object:

```
decision_forest::train(queue, desc, input)
  └─ detail::train_ops / train_ops_dispatcher            (detail/train_ops.hpp)
       └─ backend::train_kernel_gpu<Float, method::hist, Task>::operator()
                                                          (backend/gpu/train_kernel.hpp)
```

`train_kernel_gpu` is specialized per task in the thin wrappers:

- `train_kernel_cls_hist_dpc.cpp` — classification
- `train_kernel_reg_hist_dpc.cpp` — regression

Each wrapper picks the `Float` type (`float`/`double`) and constructs the real
worker, then calls it:

```
train_kernel_gpu<Float, method::hist, Task>::operator()
  └─ call_train_kernel<Float>(...)
       └─ train_kernel_hist_impl<Float, Bin=uint32, Index=int32, Task> impl(ctx);
          return impl(desc, data, responses, weights);   // operator()
```

`train_kernel_hist_impl` (declared in `train_kernel_hist_impl.hpp`, defined in
`train_kernel_hist_impl_dpc.cpp`) is where all training logic lives. Its
`operator()` is the true driver.

---

## 2. Top-level driver: `train_kernel_hist_impl::operator()`

Defined in `train_kernel_hist_impl_dpc.cpp`. High-level structure:

```
operator()(desc, data, responses, weights)
  ├─ validate_input(desc, data, responses)
  ├─ init_params(ctx, desc, data, responses, weights)     // fill train_context
  ├─ allocate_buffers(ctx)
  ├─ model_manager(ctx, tree_count, column_count)
  ├─ device_engine (RNG)
  │
  └─ for (iter = 0; iter < tree_count; iter += tree_in_block)   // TREE-BLOCK loop
       ├─ init root node_list for the block (one root per tree)
       ├─ gen_initial_tree_order(...)                     // bootstrap / row sampling
       ├─ compute_initial_histogram(...)                  // ROOT stats (level 0)
       ├─ [oob] get_oob_row_list(...)
       │
       ├─ for (level = 0; node_count > 0; ++level)         // PER-LEVEL loop
       │    ├─ gen_feature_list(...)                       // random feature subset / node
       │    ├─ gen_random_thresholds(...)                  // only used by random splitter
       │    ├─ compute_best_split(...)                     // choose best split / node
       │    ├─ tree_level_record(...)                      // snapshot level to host
       │    ├─ [mdi] update_mdi_var_importance(...)
       │    ├─ get_split_node_count(...)                   // how many nodes actually split
       │    └─ if (split nodes) :
       │         ├─ [distr] calculate_left_child_row_count_on_local_data(...)
       │         ├─ do_node_split(...)                     // build children (parent → L/R)
       │         └─ do_level_partition_by_groups(...)      // reorder rows by child
       │
       ├─ model_manager.add_tree_block(level_records, ...)
       └─ for each tree in block : compute_results(...)    // oob err, var importance
  │
  └─ finalize_oob_error(...) / finalize_var_imp(...)  →  train_result
```

Two nested loops define everything:

- **Tree-block loop** (`iter`): trees are trained in blocks of `tree_in_block`.
  Each block starts from a fresh set of root nodes (one root per tree).
- **Per-level loop** (`level`): breadth-first tree growth. At each level every
  live node is evaluated, split, and its children become the next level.

---

## 3. `init_params` — the training context

`init_params` populates `train_context` (`train_misc_structs.hpp`) from the
descriptor and data. Notable fields for this discussion:

- `is_weighted_` — true iff a weights table with one weight per row was provided.
- `min_observations_in_leaf_node_` — min sample count per child.
- `min_weight_leaf_` — min summed weight per child (weighted training only).
- `splitter_mode_value_` — `best` (exhaustive over bins) or `random` (ExtraTrees).
- `distr_mode_` — multi-rank distributed training.

**Sample-weight / `min_weight_fraction_in_leaf_node` handling** (in `init_params`):

- Weighted training: `min_weight_leaf_ = min_weight_fraction * (total dataset weight)`
  (a single scalar reduction over the weights, matching the DAAL CPU convention).
- Unweighted training + non-zero fraction: no fake weights are used. Instead the
  fraction is folded into the sample-count constraint:
  `min_observations_in_leaf_node_ = max(min_observations_in_leaf_node_,
  ceil(min_weight_fraction * n_samples))`, and `min_weight_leaf_` stays 0.

---

## 4. Per-node data layout

Two parallel structures hold per-node state at each level.

### `node_list` — integer node properties (`train_node_helpers.hpp`)
An `Index` array, `node_prop_count_ = 8` slots per node. Relevant indices:

| index | name        | meaning                          |
|-------|-------------|----------------------------------|
| 0     | `ind_ofs`   | local row offset into tree_order |
| 1     | `ind_lrc`   | local row count                  |
| 2     | `ind_fid`   | split feature id (leaf_mark if leaf) |
| 3     | `ind_bin`   | split bin threshold              |
| 4     | `ind_lch_grc` | left child global row count     |
| 5     | `ind_win`   | winner class (classification)    |
| 6     | `ind_grc`   | global row count                 |
| 7     | `ind_lch_lrc` | left child local row count (distr) |

### `impurity_data` — float/hist per-node stats (`train_impurity_data.hpp`)
Because `node_list` is integer-typed, all floating stats live here:

- `imp_list_` — per-node impurity. Classification: `[gini]`. Regression:
  `[mean, sum2cent]` (extensive sum of squared deviations).
- `class_hist_list_` — per-node class counts (classification only).
- `node_weight_list_` — per-node **total summed sample weight** (used for weighted
  impurity decrease and the `min_weight_leaf` check). Propagated like row count.

`imp_data_list_ptr` / `imp_data_list_ptr_mutable` bundle raw device pointers to
the above so kernels can read/write them.

---

## 5. Root node statistics (level 0)

Function chain, called **once per tree block**, before the per-level loop:

```
compute_initial_histogram(ctx, response, weights, tree_order, node_list, imp_data, ...)
  ├─ (non-distr, both tasks) compute_initial_histogram_local(...)
  │     └─ per node, per row-slice: compute_hist_for_node(...)   // device function
  │            classification: builds class_hist_list_ (atomic add),
  │                            node gini into imp_list_,
  │                            + sums weights into node_weight_list_ (atomic add)
  │            regression:     builds mean/sum2cent into imp_list_,
  │                            + reduces weights into node_weight_list_
  └─ (distr) task-specific: histogram/sum/sum2cent kernels + allreduce, then
             compute_initial_imp_for_node_list / fin_initial_imp
```

`compute_hist_for_node` is the only place a node's stats are built by scanning
rows from scratch. The root's **total weight** is computed here (piggybacking on
the row loop that already builds the histogram); for unweighted training it is
seeded with the row count. Thereafter node weight is only *propagated*, never
re-scanned.

---

## 6. Per-level splitting: `compute_best_split`

For every live node at the current level, pick the best split among a random
subset of features. `compute_best_split` dispatches by splitter mode
(`train_splitter_impl_dpc.cpp`, class `train_splitter_impl`):

```
compute_best_split(...)
  ├─ splitter_mode::best   → train_splitter_impl::best_split(...)
  └─ otherwise (random)    → train_splitter_impl::random_split(...)
```

### `best_split` (histogram splitter)
Two SYCL kernels:

1. **Main kernel** — for each (node, feature): `compute_histogram(...)` builds a
   per-bin cumulative histogram in local memory (`hist[bin]` = stats of rows with
   `value ≤ bin`), plus `local_weight[bin]` = summed weight of the left side. For
   each candidate bin it forms a `split_info`, calls `calc_imp_dec` to score the
   split, and keeps the best per feature via `test_split_is_best`.
2. **Merge kernel** — reduces the per-feature best splits to a single best split
   per node, writes split info into `node_list` (`update_node_bs_info`) and the
   left-child stats into `left_child_imp_data` (`update_left_child_imp`).

### `random_split` (ExtraTrees splitter)
No reusable histogram: for each (node, feature) it picks one random threshold in
`[min_bin, max_bin]` and reduces over the node's rows to get left count, left
class hist / regression stats, and (if weighted) left weight sum, then scores it
the same way (`calc_imp_dec`, `choose_best_split`).

### Split scoring — `calc_imp_dec` (`train_splitter_helpers.hpp`)
Computes the impurity **decrease** used to rank splits. Both tasks use the same
structure: children weighted by size, normalized by the node total.

- Classification (intensive Gini `left_imp`, `right_imp`, `node_imp`):
  - weighted:   `imp_dec = node_imp − (W_left·left_imp + W_right·right_imp) / W_node`
  - unweighted: `imp_dec = node_imp − (n_left·left_imp + n_right·right_imp) / n_node`
- Regression (intensive MSE = `sum2cent / count`):
  - weighted:   `imp_dec = node_imp − (W_left·left_imp + W_right·right_imp) / W_node`
  - unweighted: `imp_dec = node_imp − (n_left·left_imp + n_right·right_imp) / n_node`

`W_left` comes from `local_weight` (best) or a row reduction (random);
`W_node = node_weight_list_[node]` (propagated); `W_right = W_node − W_left`.
Note child impurities are still computed from **unweighted** counts — only the
decrease is weighted (see the `TODO` in `calc_imp_dec`; full sklearn parity would
require weighting the histograms themselves).

### Split acceptance — `test_split_is_best`
A split is valid only if `imp_dec > 0` and **both** children satisfy the leaf
constraints: `count ≥ min_observations_in_leaf_node` and
`weight_sum ≥ min_weight_leaf` (the weight checks are no-ops when
`min_weight_leaf == 0`).

---

## 7. Building children: `do_node_split` / `do_node_imp_split`

Once best splits are chosen, `do_node_split` creates the next level's nodes.
Children are derived from the parent **without re-scanning rows**:

- Integer props (`do_node_split` kernel):
  `node_lch[ind_grc] = node_prn[ind_lch_grc]`,
  `node_rch[ind_grc] = node_prn[ind_grc] − node_lch[ind_grc]`.
- Float stats (`do_node_imp_split`, device function):
  - class hist / regression stats: right = parent − left (`sub_stat`);
  - node weight: `left = W_left` (stored at split time),
    `right = parent_weight − W_left` — the exact analogue of the row-count split.

Then `do_level_partition_by_groups` reorders `tree_order` so each child's rows are
contiguous, ready for the next level's histogram build.

---

## 8. Results and finalization

- `tree_level_record` snapshots each level's `node_list` + `impurity_data` to host;
  `model_manager.add_tree_block` turns the recorded levels into the output trees
  (leaf response = winner class / regression mean, leaf class histogram, impurity).
- `compute_results` computes out-of-bag error and MDA/MDI variable importance per
  tree; `finalize_oob_error` / `finalize_var_imp` aggregate across trees into the
  final `train_result`.

---

## 9. Where each quantity lives (summary)

| quantity | produced at root | per-level split | propagated to children |
|---|---|---|---|
| row count (`ind_grc`) | node_list init | — | `do_node_split` (parent−left) |
| class hist / gini | `compute_hist_for_node` | left from `best_split`/`random_split` | `do_node_imp_split` (parent−left) |
| regression mean/sum2cent | `compute_hist_for_node` | left from splitter | `do_node_imp_split` (`sub_stat`) |
| **node total weight** | `compute_hist_for_node` | `W_left` from splitter | `do_node_imp_split` (parent−left) |
| impurity decrease | — | `calc_imp_dec` | not propagated (recomputed) |

The design principle for weights mirrors row counts: the total is measured once at
the root, `W_left` is read from the histogram/reduction already computed for each
split, and `W_right = W_node − W_left` is propagated — no extra per-level scan on
the histogram (`best`) path.
