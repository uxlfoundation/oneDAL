<!--
  ~ Copyright contributors to the oneDAL project
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# The tasks, and why they look like this

Each directory here is one job a coding agent is given on a copy of oneDAL. The grader decides by rebuilding and
rerunning oneDAL tests, not by asking a model. The point is to compare *arms*: the same tasks with and without
the repository's agent guidance (`AGENTS.md`, `.github/instructions`), or with other tooling or layout. A change to
the guidance can then be judged by what agents do with it.

The key format is in [`../README.md`](../README.md#tasks). This page explains the choices.

## What every task has in common

- **A real oneDAL commit as the starting point (`base`).** The agent gets a copy of the tree at that commit, with
  no git history and without `dev/agent_evals`. With history it could read the fix, and with this directory it
  could read the answer.
- **A hidden check the agent never sees.** For a fix it is the test that came with the upstream fix, or one written
  for the task. The grader copies it into the agent's tree only while grading, then removes it.
- **Pass means more than "the new test passes".** Fix and feature tasks are gated on `strict_pass`: the hidden
  tests pass *and* the existing tests of the same component (`regression_targets`) still pass. A fix that breaks
  something else does not count.
- **CPU only, no oneAPI compilers.** Targets are `*_host` Bazel tests, so they build with gcc on a plain Linux box
  or a CI runner. Several prompts say so, because otherwise agents spend their time trying to install `icpx`.
- **Self-checked.** `run.py check` proves for every task that the untouched tree fails and the reference solution
  passes. A task that cannot show both is broken, however reasonable it looks.

## Families

### build (3): can the agent find and run the right build?

| task | asked | graded | why it is here |
|---|---|---|---|
| `build_kmeans_host` | build and run the CPU K-Means batch tests with Bazel, report the command | the grader reruns the reported command and needs `//cpp/oneapi/dal/algo/kmeans:test_batch_host` to show `PASSED` | target names are the first thing guidance has to teach; the agent must find a target it can run on this host |
| `build_release_host` | build the CPU release package with Bazel, report the path of `libonedal_core` | the path must be a real ELF file inside the release layout of this run | tests whether the agent knows the release target and layout, not just `bazel build //...` |
| `build_make_gnu` | build the DAAL CPU libraries with **make** (oneMKL/oneTBB preinstalled under `/opt/onedal-make-deps`) | `libonedal_core.so` under `__release_lnx_gnu`, and the run is rejected if Bazel was used | the make build is the older, less documented path; agents tend to fall back to Bazel, which is what the task forbids |

### fix (11) and feature (2): change the library so that a hidden test passes

Nine are **mined**: a merged upstream PR fixed the bug or added the feature, the task starts at its parent commit
(`base` = `<fix>^`), and the PR's own test is the hidden test. These are real oneDAL problems with a maintainer's
solution as the reference.

| task | upstream fix | what the agent must do |
|---|---|---|
| `fix_kmeans_empty_reloc` | #3742 | relocate empty K-Means clusters the way scikit-learn does (dense and CSR) |
| `fix_kmeans_csr_obj` | #3681 | correct the objective function value of sparse (CSR) K-Means |
| `fix_df_fraction_gt1` | #3720 | allow `observations_per_tree_fraction` above 1.0 in decision forest |
| `fix_df_bestfirst` | #3772 | use the right child impurity for best-first leaf order (`max_leaf_nodes`) |
| `fix_df_extratrees_reg` | #3649 | fix the random splitter of decision forest regression (ExtraTrees) |
| `fix_pca_online_signflip` | #2705 | apply the deterministic sign flip in online CPU PCA |
| `fix_table_csr_accessor` | #2421 | make `csr_accessor::pull` return consistent blocks across zero/one-based indexing |
| `fix_table_csr_zero_based` | #3347 | fix the crash on zero-based `csr_table` input in K-Means and BasicStatistics |
| `feat_cosine_self` | #3712 | let `cosine_distance` compute self-distances from a single table |

The prompts are written as bug reports or feature requests (what a user would file), not as hints about the fix.
Several upstream fixes are newer than some models' training data and some are older; reports can split results by
the base date for that reason.

Four are **constructed**. They cover what mined fixes do not: one-character bugs, CPU-dispatch code, and changes
that cross the oneAPI → DAAL layers.

| task | how it is built | why |
|---|---|---|
| `fix_avx2_dispatch` | `setup.patch` breaks the AVX2 specialisation of `popcnt64` in subgraph isomorphism (counts 32 bits, not 64); the hidden test is built with `--cpu=all` | the bug only exists in one ISA branch, so the agent has to understand oneDAL's CPU dispatch to find it |
| `fix_finiteness_mut` | `setup.patch` flips one `&&` to `\|\|` in the DAAL double-precision finiteness check | a minimal, local bug in DAAL code; `root_cause_fixed` also checks that the agent fixed that line, not a symptom elsewhere |
| `feat_xparam_cov_bias` | `setup.patch` removes the covariance `bias` option from the oneAPI interface, the DAAL parameter and the kernel; the hidden tests are the original covariance tests | a feature that must be threaded through both layers, the most common shape of real oneDAL changes |
| `fix_cel_thread_race` | a real, unfixed bug (#3820): DAAL `cross_entropy_loss` returns different results on 4+ threads; the reference fix (`oracle.patch`) and the hidden threading test are written for the task | a concurrency bug in DAAL threading code, which no mined task covers |

`fix_cel_thread_race` only fails on hardware where the race can happen. It was measured to need at least 3 CPUs,
as oneTBB counts them (the affinity mask and the cgroup quota), and it was never seen on AMD EPYC 7763. The task
therefore declares `min_cpus: 4` and `cpu_vendors: ["GenuineIntel"]`; on other hosts the check reports `SKIPPED`
instead of a false pass (see `graders/fix_hidden.py`).

### test (2): write tests that would catch bugs

| task | asked | graded |
|---|---|---|
| `test_add_chebyshev` | add CPU tests for the Chebyshev distance used by brute-force kNN | **mutation testing**: the agent's tests must pass on the real code and fail on at least 60% of 7 seeded bugs (`mutants/*.patch`: missing `abs`, sum instead of max, skipped feature, wrong result index, ...) |
| `test_add_minkowski` | the same for Minkowski distance and its `degree` parameter | the same, with 7 mutants (degree ignored, `p=0` accepted, square root instead of `p`-th root, ...) |

"The test passes" says nothing about a test, so the grader asks whether it would have caught real mistakes. Library
sources are restored from `base` before the tests run, so tests cannot pass by changing the code under test.

### review (9): find what a maintainer would find

The agent reviews the HEAD commit (`review_patch`) and returns findings as JSON (file, line, severity, rule,
message).

| task | the change under review | graded on |
|---|---|---|
| `review_seeded_fc` | a finiteness-checker change with 8 planted problems: 5 against the coding guidelines (header guard, missing licence, raw `new`, exception in DAAL, ...), 1 maintainer convention, 2 plain bugs (`cout`, off-by-one) | recall of the planted problems, plus any of 5 **decoys** flagged |
| `review_guidance_decoys` | a covariance change with 6 planted problems and 6 decoys: code that breaks the literal text of the guidance but is normal DAAL practice (`PRAGMA_OMP_SIMD`, `TArray`, leading-underscore members, ...) | recall, and whether the guidance makes the agent flag the decoys |
| `review_clean_control` | PR #3775 as merged: approved by a maintainer, with no problem left | pass only if the agent reports no blocker/major finding; this measures false alarms |
| `review_replay_pr3580`, `_pr3659`, `_pr3665`, `_pr3674`, `_pr3718`, `_pr3773` | real oneDAL PRs at the commit that was reviewed | recall against the threads human reviewers actually opened (`gold.json`, 4–13 threads each), matched by file and line within 5 lines |

Planted problems give a known answer. Replays give the real distribution of what maintainers comment on, but
humans also miss things and comment on non-defects, so replay recall is a **score** (`score: recall`), not
pass/fail. Decoys exist because guidance can hurt: if an arm with guidance flags more decoys, the guidance is too
literal.

## What the pilots showed (haiku, Claude Code 2.1.286, 3 runs per arm)

These numbers are from one small run and are not significant; they show which tasks discriminate.

- **Too easy for comparisons** (6/6 in both arms): `build_kmeans_host`, `build_release_host`,
  `feat_xparam_cov_bias`, `fix_avx2_dispatch`, `fix_df_fraction_gt1`, `fix_finiteness_mut`, `fix_kmeans_csr_obj`.
  They stay as a floor and to catch regressions in the environment, and they can separate weaker models.
- **Where arms and models differ:** `build_make_gnu` (3/6), `fix_df_bestfirst` (3/6), `fix_kmeans_empty_reloc`
  (4/6), `fix_table_csr_zero_based` (2/6), `feat_cosine_self` (5/6), and every review task.
- **Too hard for haiku** (0/6): `fix_cel_thread_race`, `fix_df_extratrees_reg`, `test_add_minkowski`
  (`test_add_chebyshev` 1/6). These need a stronger model to say anything about guidance.
- **Review replay recall is low for every arm** (0.0–0.26 per task). The planted-defect task is higher (0.33 without
  guidance, 0.62 with it).
- `fix_pca_online_signflip` and `fix_table_csr_accessor` have no guidance file at their base, so the arms are
  identical there; they are left out of arm comparisons.

## Adding a task

1. Pick a base commit and a check the untouched tree fails, preferably the test of a merged fix.
2. Write the prompt as the user or the reporter would, without the solution.
3. Add `regression_targets` for the component, and `"gate": ["strict_pass"]` for fix and feature tasks.
4. Run `python3 dev/agent_evals/run.py validate` and `python3 dev/agent_evals/run.py check <task>` until the
   untouched tree fails and the reference passes. CI runs the same check for changed tasks.
